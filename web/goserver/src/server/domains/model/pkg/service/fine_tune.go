package service

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	fp "path/filepath"
	"regexp"
	"strconv"
	"strings"

	"go.mongodb.org/mongo-driver/bson/primitive"
	"gopkg.in/yaml.v2"

	assetFindOne "server/db/pkg/handler/asset/find_one"
	buildFindOne "server/db/pkg/handler/build/find_one"
	cvatTaskFind "server/db/pkg/handler/cvat_task/find"
	modelFindOne "server/db/pkg/handler/model/find_one"
	modelInsertOne "server/db/pkg/handler/model/insert_one"
	modelUpdateOne "server/db/pkg/handler/model/update_one"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	t "server/db/pkg/types"
	splitState "server/db/pkg/types/build/split_state"
	modelStatus "server/db/pkg/types/status/model"
	kitendpoint "server/kit/endpoint"
	"server/kit/utils/basic/arrays"
	ufiles "server/kit/utils/basic/files"
	trainingWorkerGpuNum "server/workers/train/pkg/handler/get_gpu_amount"
	runCommandsWorker "server/workers/train/pkg/handler/run_commands"
)

type FineTuneRequestData struct {
	BatchSize              int    `json:"batchSize"`
	BuildId                string `json:"buildId"`
	Epochs                 int    `json:"epochs"`
	GpuNum                 int    `json:"gpuNumber"`
	Name                   string `json:"name"`
	ParentModelId          string `json:"parentModelId"`
	ProblemId              string `json:"problemId"`
	SaveAnnotatedValImages bool   `json:"saveAnnotatedValImages"`
}

func (s *basicModelService) FineTune(ctx context.Context, req FineTuneRequestData) chan kitendpoint.Response {
	returnChan := make(chan kitendpoint.Response)
	go func() {
		defer close(returnChan)
		parentModel, build, problem := s.getParentModelBuildProblem(req.ParentModelId, req.BuildId, req.ProblemId)
		trainingWorkDir, _ := createTrainingWorkDir(problem.WorkingDir, req.Name)
		gpuNum := s.getOptimalGpuNumber(req.GpuNum, parentModel.TrainingGpuNum)
		newModel, err := s.createNewModel(ctx, req.Name, problem, parentModel.Id, trainingWorkDir, gpuNum)
		if err != nil {
			returnChan <- kitendpoint.Response{Data: nil, Err: err, IsLast: true}
			return
		}
		commands, newModelOutput := s.prepareFineTuneCommands(newModel.Scripts.Train, req.BatchSize, gpuNum, req.Epochs, newModel, parentModel, build, problem, req.SaveAnnotatedValImages)
		// TODO: run eval script
		outputLog := fmt.Sprintf("%s/output.log", newModel.Dir)
		env := getFineTuneEnv()
		s.runCommand(commands, env, newModel.TrainingWorkDir, outputLog)
		copySnapshotLatestToModelPath(newModel.TrainingWorkDir, newModel.SnapshotPath)
		copyConfigToModelPath(newModel.TrainingWorkDir, newModel.ConfigPath)
		newModel = s.saveModelMetrics(newModelOutput, build.Id, newModel)

		returnChan <- kitendpoint.Response{Data: newModel, Err: nil, IsLast: true}
	}()
	return returnChan
}

func getFineTuneEnv() []string {
	return []string{}
}

func (s *basicModelService) saveModelMetrics(output string, buildId primitive.ObjectID, model t.Model) t.Model {
	newModelYamlFile, err := ioutil.ReadFile(output)
	if err != nil {
		log.Println("ReadFile", err)
	}
	var metrics struct {
		Metrics []t.Metric `yaml:"metrics"`
	}
	err = yaml.Unmarshal(newModelYamlFile, &metrics)
	if err != nil {
		log.Println("Unmarshal", err)
	}
	model.Metrics = make(map[string][]t.Metric)
	model.Metrics[buildId.Hex()] = metrics.Metrics
	model.Status = modelStatus.Finished

	modelUpdateOneRes := <-modelUpdateOne.Send(
		context.TODO(),
		s.Conn,
		model,
	)
	return modelUpdateOneRes.Data.(modelUpdateOne.ResponseData)
}

func (s *basicModelService) prepareFineTuneCommands(script string, batchSize, gpuNum, epochs int, model, parentModel t.Model, build t.Build, problem t.Problem, saveAnnotatedValImages bool) ([]string, string) {
	configPyFile, err := os.Open(parentModel.ConfigPath)
	if err != nil {
		fmt.Println("fine_tune.prepareFineTuneCommands.os.Open(parentModel.ConfigPath)", err)
	}
	defer configPyFile.Close()
	byteValue, err := ioutil.ReadAll(configPyFile)
	if err != nil {
		log.Println("fine_tune.prepareFineTuneCommands.ioutil.ReadAll(configPyFile)", err)
	}
	defaultTotalEpochs := getVarValInt(byteValue, "total_epochs")
	trainImgPrefixes, trainAnnFiles := s.getImgPrefixAndAnnotation("train", build, problem)
	valImgPrefixes, valAnnFiles := s.getImgPrefixAndAnnotation("val", build, problem)
	testImgPrefixes, testAnnFiles := s.getImgPrefixAndAnnotation("test", build, problem)
	classes, numClasses := getClasses(problem.Labels)
	updateConfigArr := []string{
		fmt.Sprintf("total_epochs=%d", defaultTotalEpochs+epochs),
		fmt.Sprintf("resume_from=%s", parentModel.SnapshotPath),
		fmt.Sprintf("work_dir=%s", model.TrainingWorkDir),
		fmt.Sprintf("data.train.dataset.ann_file=%s", strings.Join(trainAnnFiles, ",")),
		fmt.Sprintf("data.train.dataset.img_prefix=%s", strings.Join(trainImgPrefixes, ",")),
		fmt.Sprintf("data.train.dataset.classes=(%s,)", classes),
		fmt.Sprintf("data.val.ann_file=%s", strings.Join(valAnnFiles, ",")),
		fmt.Sprintf("data.val.img_prefix=%s", strings.Join(valImgPrefixes, ",")),
		fmt.Sprintf("data.val.classes=(%s,)", classes),
		fmt.Sprintf("data.test.ann_file=%s", strings.Join(testAnnFiles, ",")),
		fmt.Sprintf("data.test.img_prefix=%s", strings.Join(testImgPrefixes, ",")),
		fmt.Sprintf("data.test.classes=(%s,)", classes),
		fmt.Sprintf("model.bbox_head.num_classes=%d", numClasses),

		fmt.Sprintf("--resume-from %s", parentModel.SnapshotPath),
		fmt.Sprintf("--train-ann-files %s", strings.Join(trainAnnFiles, ",")),
		fmt.Sprintf("--train-data-roots %s", strings.Join(trainImgPrefixes, ",")),
		fmt.Sprintf("--val-ann-files %s", strings.Join(valAnnFiles, ",")),
		fmt.Sprintf("--val-data-roots %s", strings.Join(valImgPrefixes, ",")),
		fmt.Sprintf("--save-checkpoints-to %s", model.TrainingWorkDir),
		fmt.Sprintf("--epochs %s", defaultTotalEpochs+epochs),
	}

	if batchSize > 0 {
		updateConfigArr = append(updateConfigArr, fmt.Sprintf("data.samples_per_gpu=%d", batchSize))
	} else {
		updateConfigArr = append(updateConfigArr, fmt.Sprintf("data.samples_per_gpu=auto"))
	}

	updateConfigStr := strings.Join(updateConfigArr, " ")
	output := fp.Join(model.Dir, "model.yml")
	err = os.Chmod(script, 0777)
	if err != nil {
		log.Println("Chmod", err)
	}
	fineTuneCommand := fmt.Sprintf(`python3 %s %s %d %s --update_config="%s"`, script, parentModel.ConfigPath, gpuNum, output, updateConfigStr)
	if saveAnnotatedValImages {
		annotatedValImagesDir := fp.Join(model.Dir, "valImages")
		fineTuneCommand = strings.Join([]string{
			fineTuneCommand,
			fmt.Sprintf("--show-dir=%s", annotatedValImagesDir),
		}, " ")
	}
	commands := []string{
		fineTuneCommand,
	}
	for _, c := range commands {
		log.Println(c)
	}
	return commands, output
}

func (s *basicModelService) getOptimalGpuNumber(gpuNum, parentGpuNum int) int {
	workerGpuNum := s.getTrainingWorkerGpuNum()
	log.Printf("getOptimalGpuNumber worker = %d, parent = %d, fromUI = %d", workerGpuNum, parentGpuNum, gpuNum)
	gpus := []int{gpuNum, parentGpuNum, workerGpuNum}
	gpus = arrays.FilterInt(gpus, func(a int) bool {
		return a > 0
	})
	result := arrays.MinInt(gpus)
	log.Printf("getOptimalGpuNumber result = %d", result)
	return result
}

func (s *basicModelService) getTrainingWorkerGpuNum() int {
	trainingWorkerGpuNumResp := <-trainingWorkerGpuNum.Send(
		context.TODO(),
		s.Conn,
		trainingWorkerGpuNum.RequestData{},
	)
	return trainingWorkerGpuNumResp.Data.(trainingWorkerGpuNum.ResponseData).Amount
}

func (s *basicModelService) runCommand(commands, env []string, workingDir, outputLog string) {
	<-runCommandsWorker.Send(
		context.Background(),
		s.Conn,
		runCommandsWorker.RequestData{
			Commands:  commands,
			OutputLog: outputLog,
			WorkDir:   workingDir,
			Env:       env,
		},
	)
}

func (s *basicModelService) createNewModel(
	ctx context.Context,
	name string,
	problem t.Problem,
	parentModelId primitive.ObjectID,
	trainingWorkDir string,
	gpuNum int,
) (t.Model, error) {
	dir := fp.Join(problem.Dir, name)
	if err := os.MkdirAll(dir, 0777); err != nil {
		log.Println("evaluate.createFolder.os.MkdirAll(path, 0777)", err)
	}
	tensorBoardLogDir := fp.Join(problem.WorkingDir, name)
	snapshotPath := fp.Join(dir, "snapshot.pth")
	configPath := fp.Join(dir, "model.py")
	newModelResp := <-modelInsertOne.Send(
		ctx,
		s.Conn,
		modelInsertOne.RequestData{
			ConfigPath:        configPath,
			Dir:               dir,
			Name:              name,
			ParentModelId:     parentModelId,
			ProblemId:         problem.Id,
			SnapshotPath:      snapshotPath,
			Status:            modelStatus.InProgress,
			TensorBoardLogDir: tensorBoardLogDir,
			TrainingGpuNum:    gpuNum,
			TrainingWorkDir:   trainingWorkDir,
		},
	)
	model := newModelResp.Data.(modelInsertOne.ResponseData)
	err := newModelResp.Err
	if err != nil {
		return model, newModelResp.Err.(error)
	} else {
		return model, nil
	}
}

func (s *basicModelService) getParentModelBuildProblem(modelIdString, buildIdString, problemIdString string) (t.Model, t.Build, t.Problem) {
	parentModelId, err := primitive.ObjectIDFromHex(modelIdString)
	if err != nil {
		log.Fatalln("ObjectIDFromHex", err)
	}
	buildId, err := primitive.ObjectIDFromHex(buildIdString)
	if err != nil {
		log.Fatalln("ObjectIDFromHex", err)
	}
	problemId, err := primitive.ObjectIDFromHex(problemIdString)
	if err != nil {
		log.Fatalln("ObjectIDFromHex", err)
	}
	chParentModel := modelFindOne.Send(
		context.TODO(),
		s.Conn,
		modelFindOne.RequestData{Id: parentModelId},
	)
	chBuild := buildFindOne.Send(
		context.TODO(),
		s.Conn,
		buildFindOne.RequestData{Id: buildId},
	)
	chProblem := problemFindOne.Send(
		context.TODO(),
		s.Conn,
		problemFindOne.RequestData{Id: problemId},
	)
	rParentModel, rBuild, rProblem := <-chParentModel, <-chBuild, <-chProblem
	parentModel := rParentModel.Data.(modelFindOne.ResponseData)
	build := rBuild.Data.(buildFindOne.ResponseData)
	problem := rProblem.Data.(problemFindOne.ResponseData)

	return parentModel, build, problem
}

func createTrainingWorkDir(problemWorkingDir, newModelName string) (string, error) {
	trainingWorkDir := fp.Join(problemWorkingDir, newModelName)
	log.Println("Create Dir:", trainingWorkDir)
	err := os.MkdirAll(trainingWorkDir, 0777)
	if err != nil {
		log.Println("MkdirAll", trainingWorkDir, err)
		return "", err
	}
	return trainingWorkDir, nil
}

func (s *basicModelService) createNewModelDir(problemFolder, newModelName string) string {
	newModelDirPath := fp.Join(problemFolder, newModelName)
	if err := os.Mkdir(newModelDirPath, 0777); err != nil {
		log.Println("fine_tune.createNewModelDir.os.Mkdir(newModelDirPath, 0777)", err)
	}
	return newModelDirPath
}

func (s *basicModelService) createNewBuildConfig(
	newModelDirPath, trainingWorkDir string,
	epochs int,
	parentModel t.Model,
	build t.Build,
	problem t.Problem,
) (string, error) {
	newModelConfigPath := fmt.Sprintf("%s/config.py", newModelDirPath)
	configPyFile, err := os.Open(parentModel.ConfigPath)
	if err != nil {
		fmt.Println("configPyFile", err)
	}
	defer configPyFile.Close()
	byteValue, _ := ioutil.ReadAll(configPyFile)

	defaultTotalEpochs := getVarValInt(byteValue, "total_epochs")
	byteValue = replaceVarValInt(byteValue, "total_epochs", defaultTotalEpochs+epochs)
	byteValue = replaceVarValStr(byteValue, "resume_from", parentModel.SnapshotPath)
	byteValue = replaceVarValStr(byteValue, "work_dir", trainingWorkDir)

	trainImgPrefixes, trainAnnFiles := s.getImgPrefixAndAnnotation("train", build, problem)
	byteValue = replaceVarValList(byteValue, "train_ann_file", stringArrToString(trainAnnFiles))
	byteValue = replaceVarValList(byteValue, "train_img_prefix", stringArrToString(trainImgPrefixes))

	valImgPrefixes, valAnnFiles := s.getImgPrefixAndAnnotation("val", build, problem)
	byteValue = replaceVarValList(byteValue, "val_ann_file", stringArrToString(valAnnFiles))
	byteValue = replaceVarValList(byteValue, "val_img_prefix", stringArrToString(valImgPrefixes))

	testImgPrefixes, testAnnFiles := s.getImgPrefixAndAnnotation("test", build, problem)
	byteValue = replaceVarValList(byteValue, "test_ann_file", stringArrToString(testAnnFiles))
	byteValue = replaceVarValList(byteValue, "test_img_prefix", stringArrToString(testImgPrefixes))

	err = ioutil.WriteFile(newModelConfigPath, byteValue, 0777)
	if err != nil {
		fmt.Println("WriteFile", err)
		return newModelConfigPath, err
	}
	return newModelConfigPath, nil
}

func (s *basicModelService) getImgPrefixAndAnnotation(category string, build t.Build, problem t.Problem) ([]string, []string) {
	trainAssetIds := getAssetsIdsList(category, build.Split["."].Children)

	cvatTaskFindResp := <-cvatTaskFind.Send(context.TODO(), s.Conn, cvatTaskFind.RequestData{ProblemId: problem.Id, AssetIds: trainAssetIds})
	cvatTasks := cvatTaskFindResp.Data.(cvatTaskFind.ResponseData).Items
	buildPath := fmt.Sprintf("%s/_builds/%s", problem.Dir, build.Folder)

	var annFiles []string
	var imgPrefixes []string
	for _, cvatTask := range cvatTasks {
		annFileName := fmt.Sprintf("%s.json", strconv.Itoa(cvatTask.Annotation.Id))
		annFilePath := fmt.Sprintf("%s/%s", buildPath, annFileName)
		annFiles = append(annFiles, annFilePath)
		asset := s.getAsset(cvatTask.AssetId)
		imgPrefixes = append(imgPrefixes, asset.CvatDataPath)
	}
	if len(annFiles) != len(imgPrefixes) {
		log.Println("domains.model.pkg.service.fine_tune.getImgPrefixAndAnnotation: len(annFiles) != len(imgPrefixes)")
	}
	return imgPrefixes, annFiles
}

func (s *basicModelService) getAsset(assetId primitive.ObjectID) t.Asset {
	assetFindOneResp := <-assetFindOne.Send(context.TODO(), s.Conn, assetFindOne.RequestData{Id: assetId})
	return assetFindOneResp.Data.(assetFindOne.ResponseData)
}

func stringArrToString(arr []string) string {
	b, err := json.Marshal(arr)
	if err != nil {
		log.Println("domains.model.pkg.service.fine_tune.stringArrToString.json.Marshal(arr)", err)
		return ""
	}
	return string(b)
}

func getAssetsIdsList(category string, buildSplit map[string]t.BuildAssetsSplit) []primitive.ObjectID {
	var result []primitive.ObjectID
	for _, child := range buildSplit {
		if len(child.Children) == 0 && isCategory(child, category) {
			result = append(result, child.AssetId)
		} else {
			childAssetIdsList := getAssetsIdsList(category, child.Children)
			result = append(result, childAssetIdsList...)
		}
	}
	return result
}

func isCategory(split t.BuildAssetsSplit, category string) bool {
	category = strings.ToLower(category)
	if split.Train == splitState.Confirmed && category == "train" {
		return true
	} else if split.Test == splitState.Confirmed && category == "test" {
		return true
	} else if split.Val == splitState.Confirmed && category == "val" {
		return true
	}
	return false

}

func replaceVarValList(src []byte, name string, value string) []byte {
	pattern := fmt.Sprintf(`%s[ \t]*=[ \t]*['"]?.*['"]?[ \t]*`, name)
	r := regexp.MustCompile(pattern)
	newLine := fmt.Sprintf(`%s = %s`, name, value)
	return r.ReplaceAll(src, []byte(newLine))
}

func replaceVarValStr(src []byte, name string, value string) []byte {
	pattern := fmt.Sprintf(`%s[ \t]*=[ \t]*['"]?.*['"]?[ \t]*`, name)
	r := regexp.MustCompile(pattern)
	newLine := fmt.Sprintf(`%s = "%s"`, name, value)
	return r.ReplaceAll(src, []byte(newLine))
}

func replaceVarValInt(src []byte, name string, value int) []byte {
	pattern := fmt.Sprintf(`%s[ \t]*=[ \t]*.*[ \t]*`, name)
	r := regexp.MustCompile(pattern)
	newLine := fmt.Sprintf(`%s = %d`, name, value)
	return r.ReplaceAll(src, []byte(newLine))
}

func getVarValInt(src []byte, name string) int {
	pattern := fmt.Sprintf(`%s[ \t]*=[ \t]*(.*)[ \t]*`, name)
	r := regexp.MustCompile(pattern)
	submatch := r.FindSubmatch(src)
	intRes, err := strconv.Atoi(string(submatch[1]))
	if err != nil {
		log.Println("Atoi", err)
	}
	return intRes
}

func copySnapshotLatestToModelPath(trainingPath, newSnapshotPath string) {
	snapshotPath := fmt.Sprintf("%s/latest.pth", trainingPath)
	if _, err := ufiles.Copy(snapshotPath, newSnapshotPath); err != nil {
		log.Println("copySnapshotLatestToModelPath.Copy", err)
	}
}

func copyConfigToModelPath(trainingPath, newConfigPath string) {
	configPath := fmt.Sprintf("%s/config.py", trainingPath)
	if _, err := ufiles.Copy(configPath, newConfigPath); err != nil {
		log.Println("copyConfigToModelPath.Copy", err)
	}
}
