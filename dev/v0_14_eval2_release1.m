
%% PCB defect localization via defect-free reference (public release)
% Training-free defect localization for the PCB dataset using a defect-free reference.
% Before running (required)
%
% IMPORTANT DISCLAIMER / CITATION
% - If you use this code (or any derivative) in academic work, please cite the
%   associated paper.
% - NOTE: The associated paper is currently under review. This script version
%   is provided exclusively to reviewers during the review stage and is not the
%   polished official public release.
% - An improved and finalized public version will be prepared and released
%   later. Please do not redistribute this reviewer-stage version.
%
%  1) Download the dataset:
%       https://robotics.pkusz.edu.cn/resources/dataset/
%  2) Unzip and place it into this *relative* folder (repo root):
%       dev/PCB_DATASET/
%
% Expected dataset structure (GT XML optional):
%  - dev/PCB_DATASET/images（Defective）/<DefectClass>/*
%  - dev/PCB_DATASET/PCB_USED（Defect-free）/*
%  - dev/PCB_DATASET/Annotations/<DefectClass>/<imageBase>.xml
%
% If your filesystem uses ASCII parentheses, these alternatives are also accepted:
%  - images(Defective)
%  - PCB_USED(Defect-free)
%
% How to run
%  - In MATLAB: set Current Folder to <repo>/dev, then run:
%       v0_14_eval2_release1
%
% Outputs (written under dev/outputs/)
%  - Optional figures + evaluation export bundle (MAT/JSON/CSV), controlled by cfg.* flags.

% 1) Configuration (thresholds, flags, output settings)
%   - All settings are controlled by the defaultConfig() function below.
%   - Please change these settings before running.
cfg = defaultConfig();

% 2) Paths (dataset folders)
paths = defaultPaths();

if ~isfolder(paths.datasetRoot)
	error(['Dataset root not found: %s\n' ...
		'Please download + unzip the dataset from:\n' ...
		'  https://robotics.pkusz.edu.cn/resources/dataset/\n' ...
		'and place it at this relative path (repo root):\n' ...
		'  dev/PCB_DATASET/\n'], paths.datasetRoot);
end

% 3) Build datastores / file lists
tBuild = timingStart(cfg, 'buildDatastores');
[defectiveDS, templateFiles] = buildDatastores(paths, cfg);
timingEnd(cfg, tBuild, 'buildDatastores');

fprintf('Defective images: %d\n', numel(defectiveDS.Files));
fprintf('Defect-free templates: %d\n', numel(templateFiles));

% 4) Main run: select template -> localize -> visualize per image
tRun = timingStart(cfg, 'runPipeline');
results = runPipeline(defectiveDS, templateFiles, paths, cfg);
timingEnd(cfg, tRun, 'runPipeline');

metrics = [];
paperOut = [];

if cfg.eval.enable
	% 5) Optional evaluation: compare predicted boxes vs VOC XML boxes
	tEval = timingStart(cfg, 'evaluateDetections');
	metrics = evaluateDetections(results, paths, cfg);
	timingEnd(cfg, tEval, 'evaluateDetections');
	disp(metrics.summary);
	if isfield(cfg.eval, 'reportPerClass') && cfg.eval.reportPerClass && isfield(metrics, 'perClassSummary')
		disp(metrics.perClassSummary);
		% Optionally export per-class table for quick spreadsheet review.
		saveCSV = true;
		if isfield(cfg.eval, 'savePerClassCSV')
			saveCSV = logical(cfg.eval.savePerClassCSV);
		end
		if saveCSV
			try
				outDir = fullfile(paths.outputRoot, cfg.output.outputDirName);
				ensureDir(outDir);
				iouThr = cfg.eval.iouThreshold;
				writetable(metrics.perClassSummary, fullfile(outDir, sprintf('perclass_PRF_counts_iou%.2f.csv', iouThr)));
			catch err
				fprintf('[Warn] Failed to write per-class CSV: %s\n', err.message);
			end
		end
	end
	if isfield(metrics, 'ap') && isfield(metrics.ap, 'summary')
		disp(metrics.ap.summary);
	end
end

% 6) Evaluation figures
if isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'enable') && cfg.paperFigs.enable
	tFig = timingStart(cfg, 'paperFigures');
    fprintf(['[Info] This step might be long depends on the CPU speed, it seem to be hang but it is actually running, please wait.\n']);
    try
		paperOut = generatePaperFigures(results, paths, cfg);
	catch err
		fprintf('[Warn] paperFigures failed: %s\n', err.message);
	end
	timingEnd(cfg, tFig, 'paperFigures');
end

% 7) Export a comprehensive evaluation bundle for later analysis.
if isfield(cfg, 'export') && isfield(cfg.export, 'enable') && cfg.export.enable
	tExp = timingStart(cfg, 'exportEvaluation');
	try
		exportEvaluationReport(results, metrics, paperOut, numel(templateFiles), paths, cfg);
	catch err
		fprintf('[Warn] exportEvaluation failed: %s\n', err.message);
	end
	timingEnd(cfg, tExp, 'exportEvaluation');
end


function cfg = defaultConfig()
	cfg = struct();

	% Tag outputs with script name to avoid overwrites.
	thisScript = mfilename;
	if isempty(thisScript)
		thisScript = 'run';
	end
	thisScript = regexprep(thisScript, '[^a-zA-Z0-9_\-]+', '_');

	% ---- Processing / run control ----
	% Controls how many images to run and determinism.
	cfg.run.maxImages = 10;                 % change to [] to run all images
	cfg.run.randomSeed = 0;

	% ---- Paths ----
	% Optional shuffle of defective image order.
	cfg.paths.defectiveShuffle = true;

	% ---- Template selection (global) ----
	% Chooses the defect-free reference via SSIM (optionally after cheap preselect).
	% Working resolution is controlled by getTemplateSelectionMaxDim():
	%   - downscaleMaxDim (if set) overrides downscalePreset.
	cfg.templateSelection.downscaleMaxDim = [];      % set e.g. 96 for ultra-fast legacy mode
	cfg.templateSelection.downscalePreset = '240p';  % mapped via getTargetLongSide() presets
	cfg.templateSelection.cache = true;              % cache best template per test image
	cfg.templateSelection.cachePreprocessedTemplates = true; % cache preprocessed templates in memory
	% Disk cache: repeated runs can skip preprocessing.
	cfg.templateSelection.diskCache.enable = true;
	cfg.templateSelection.diskCache.dir = '';        % empty => defaults to <devRoot>/cache
	cfg.templateSelection.diskCache.forceRebuild = false;

	% If true, crop to ROI before scoring (faster than masking).
	cfg.templateSelection.useROICropForSSIM = true;

	% Preselect: cheap score on tiny images for all templates, then SSIM on top-K.
	cfg.templateSelection.preselect.enable = true;
	cfg.templateSelection.preselect.maxDim = 128;     % tiny working size for preselect
	cfg.templateSelection.preselect.topK = 3;         % run SSIM on these candidates
	cfg.templateSelection.preselect.metric = 'corr';  % 'corr' | 'mad'

	% Optional parallelization (Parallel Computing Toolbox).
	cfg.templateSelection.parallel.enable = false;

	cfg.templateSelection.reportTopK = 3;             % prints top-K templates per test image

	% ---- Processing resolution (speed) ----
	% Runs the localization stage on a downscaled copy (boxes mapped back).
	% Presets are resolved by getTargetLongSide().
	cfg.resize.enable = true;
	cfg.resize.preset = '720p';            % mapped via getTargetLongSide() presets
	cfg.resize.targetLongSide = 1280;      % used when preset='custom' (pixels)
	cfg.resize.scaleMaskParams = true;     % scale morphology+minArea with resolution
	cfg.resize.visualizeOnOriginal = false; % if true, show original images (slower/heavier)

	% ---- Registration (alignment) ----
	% Align test->template to reduce shift/rotation false positives.
	cfg.registration.enable = true;                % requires Image Processing Toolbox
	cfg.registration.type = 'translation';         % 'translation' | 'rigid'
	cfg.registration.fallbackToTranslation = true; % if 'rigid' fails, try translation
	cfg.registration.validateTransform = true;     % accept alignment only if it improves similarity
	cfg.registration.minSSIMGain = 0.0;            % require afterSSIM >= beforeSSIM + minSSIMGain
	cfg.registration.validationUseROI = true;      % validate SSIM only inside ROI (recommended)

	% ---- SSIM parameters (explicit) ----
	% This pipeline normalizes grayscale images into [0,1], so DynamicRange=1.
	% Note: supported name-value options can differ by MATLAB version.
	% computeSSIMSafe() will fall back to default SSIM if a parameter is unsupported.
	cfg.ssim.useExplicitParams = true;
	cfg.ssim.DynamicRange = 1;
	cfg.ssim.K1 = 0.01;
	cfg.ssim.K2 = 0.03;
	cfg.ssim.GaussianWeights = true;
	cfg.ssim.WindowSize = 11; % typical SSIM window size

	% ---- Difference score map (higher => more different) ----
	% Score map is thresholded to locate defect regions.
	% (1 - SSIM map) highlights structural changes; AbsDiff adds intensity changes.
	cfg.diff.useSSIMMap = true;
	cfg.diff.useAbsDiff = true;
	cfg.diff.absDiffWeight = 0.35;                 % score = (1-ssimMap) + w*absDiff
	cfg.diff.rescaleUsingROI = true;               % prevents border outliers from compressing ROI contrast

	% ---- Thresholding + cleanup ----
	% Quantile threshold on ROI pixels + morphology cleanup.
	cfg.mask.quantile = 0.995;                    % high quantile isolates small defects
	cfg.mask.minAreaPx = 150;                     % remove small speckles
	cfg.mask.closeRadiusPx = 6;
	cfg.mask.dilateRadiusPx = 2;

	% ---- Output ----
	% Visualization options.
	cfg.output.showFigures = false;
	cfg.output.saveFigures = false;
	% Save editable MATLAB figure files alongside exported images.
	cfg.output.saveEditableFig = true;
	cfg.output.outputDirName = sprintf('outputs_%s', thisScript);
	cfg.output.figureFormat = 'png';

	% ---- Export (comprehensive evaluation bundle) ----
	% Writes a self-contained export folder after a run (MAT/JSON/CSV), intended
	% to support later analysis and reporting.
	cfg.export.enable = true;
	% Output goes to: dev/outputs/<cfg.output.outputDirName>/<dirName>/
	% Timestamp avoids accidental overwrites across runs.
	cfg.export.dirName = sprintf('eval_export_%s_%s', thisScript, datestr(now, 'yyyymmdd_HHMMSS'));
	cfg.export.writeMat = true;
	cfg.export.writeJson = true;
	cfg.export.writeCsv = true;
	% When enabled, export helpers print warnings on file write failures.
	% Default is off to keep runs quiet (previous behavior).
	cfg.export.warnOnWriteFailures = false;
	% JSON can become very large if include per-image results/detections.
	% Keep JSON focused and store full details in .mat + CSV.
	cfg.export.jsonIncludeRawResults = false;
	cfg.export.jsonIncludeDetectionsTable = false;
	% Note: This public release does not embed external notes or write
	% repo-level artifacts.

	% ---- Paper figures ----
	% Generates evaluation figures used in analysis/reporting.
	cfg.paperFigs.enable = true;
	cfg.paperFigs.outputDirName = sprintf('paper_figures_%s', thisScript);
	cfg.paperFigs.figureFormat = 'png';
	% Save editable MATLAB .fig files alongside exported paper figures.
	cfg.paperFigs.saveEditableFig = true;
	% Optional extra vector export.
	cfg.paperFigs.exportPDF = false;
	% Presentation tuning.
	cfg.paperFigs.presentation.lowProminence = true;
	cfg.paperFigs.presentation.supplementSubdir = 'supplement';
	cfg.paperFigs.presentation.showAPInTitles = false;
	cfg.paperFigs.presentation.showIoUInTitles = false;
	% If true, embeds a small note about the IoU matching criterion.
	cfg.paperFigs.presentation.showMatchNote = false;
	% Per-class PR scaling: 'full' | 'top' | 'auto' | 'log1mP'.
	% Layout: combined legend plot or small multiples.
	cfg.paperFigs.perClassPR.scaleMode = 'auto';
	cfg.paperFigs.perClassPR.combineIntoSinglePlot = true;
	cfg.paperFigs.perClassPR.topMargin = 0.02; % ymin = min(P)-margin (clamped)
	cfg.paperFigs.perClassPR.minSpan = 0.10;   % ensure at least [1-minSpan, 1]
	% Per-class AP-vs-IoU layout.
	cfg.paperFigs.perClassAPvsIoU.combineIntoSinglePlot = true;
	% IoU thresholds to sweep for AP-vs-IoU figures.
	cfg.paperFigs.iouSweep = 0.10:0.10:0.90;
	cfg.paperFigs.iouSweepCOCO = 0.50:0.05:0.95;
	% Operating point (for error breakdown + qualitative montages).
	% - 'bestF1': chooses a score threshold maximizing global F1.
	% - numeric: fixed score threshold.
	cfg.paperFigs.operatingPoint = 'bestF1';
	% How many examples to show in qualitative montages.
	cfg.paperFigs.maxFPExamples = 9;
	cfg.paperFigs.maxFNExamples = 9;
	cfg.paperFigs.maxTPExamples = 9;
	% If true, saves *separate* montages per class (instead of one mixed montage).
	% This avoids a single frequent class dominating the qualitative figures.
	cfg.paperFigs.montage.splitByClass = true;
	% Step-by-step example selection.
	cfg.paperFigs.stepByStep.exampleMode = 'worstFP'; % 'first' | 'worstFP' | 'bestTP'
	cfg.paperFigs.stepByStep.forceIndex = [];         % set to a specific results index to override

	% ---- Evaluation ----
	% IoU-based matching between predicted and GT boxes.
	cfg.eval.enable = true;
	% Note: this pipeline produces coarse boxes; IoU is used as a matching criterion
	% (TP/FP/FN), not a tight-localization claim.
	cfg.eval.iouThreshold = 0.1;
	cfg.eval.reportPerClass = true;
	cfg.eval.reportImageLevel = true;             % also report image-level detection/recall
	cfg.eval.savePerClassCSV = true;              % whether to write per-class TP/FP/Prec/Rec/F1 CSV
	% PR-curve metric (Average Precision / mAP). Uses predicted bbox scores as confidence.
	% If iouThresholds has multiple values, we report mAP = mean(AP over thresholds).
	cfg.eval.ap.enable = true;
	cfg.eval.ap.iouThresholds = [];                % [] => uses cfg.eval.iouThreshold
	cfg.eval.ap.useVOC07 = false;                  % if true, uses VOC07 11-point AP
	% K-fold cross-validation for operating-point threshold selection/reporting.
	% Threshold is selected on calibration folds and evaluated on held-out fold.
	cfg.eval.cv.enable = true;
	cfg.eval.cv.K = 5;
	cfg.eval.cv.seed = cfg.run.randomSeed;

	% ---- Bounding box scoring ----
	% Score each connected component using a robust statistic of scoreMap pixels.
	% - 'quantile' is usually more stable than 'max' (less sensitive to hot pixels).
	cfg.bbox.scoreMethod = 'quantile';            % 'quantile' | 'mean' | 'max'
	cfg.bbox.scoreQuantile = 0.95;                % used if scoreMethod='quantile'

	% ---- Avoidance padding (ignore outer border) ----
	% The padding is defined as a fraction of the *image area* to ignore.
	% A single padPx value is computed and applied to all 4 sides.
	cfg.roi.enable = true;
	cfg.roi.avoidAreaRatio = 0.07;               % default: ignore 7% of image area

	% ---- Debug ----
	% verbose: print per-image summary including selected template and ROI padding.
	% showIntermediateMaps: include SSIM/abs-diff panels in the figure.
	cfg.debug.verbose = false;
	cfg.debug.showIntermediateMaps = false;

	% ---- Timing (profiling / bottlenecks) ----
	% Enables timestamped timing logs for major pipeline stages.
	% Kept minimal: only main stage START/END logs.
	cfg.debug.timing.enable = true;
	cfg.debug.timing.minPrintSec = 0.01; % suppress ultra-small prints if desired

end


function exportEvaluationReport(results, metrics, paperOut, numTemplates, paths, cfg)
	% Exports evaluation artifacts into a dedicated folder for later analysis.
	% Outputs (when enabled):
	%  - evaluation_report.mat : full MATLAB struct (most complete)
	%  - evaluation_report.json: compact, human-readable summary
	%  - CSVs: per-class summary, per-image summary, per-detection table, PR/FROC/AP sweeps

	if nargin < 2, metrics = []; end
	if nargin < 3, paperOut = []; end
	if nargin < 4 || isempty(numTemplates), numTemplates = NaN; end

	% Resolve export directory.
	baseOutDir = paths.outputRoot;
	if isfield(cfg, 'output') && isfield(cfg.output, 'outputDirName') && ~isempty(cfg.output.outputDirName)
		baseOutDir = fullfile(paths.outputRoot, cfg.output.outputDirName);
	end
	ensureDir(baseOutDir);

	outDirName = 'eval_export';
	if isfield(cfg, 'export') && isfield(cfg.export, 'dirName') && ~isempty(cfg.export.dirName)
		outDirName = char(cfg.export.dirName);
	end
	outDir = fullfile(baseOutDir, outDirName);
	ensureDir(outDir);

	warnOnWriteFailures = false;
	if isfield(cfg, 'export') && isfield(cfg.export, 'warnOnWriteFailures')
		warnOnWriteFailures = logical(cfg.export.warnOnWriteFailures);
	end

	% Main matching threshold.
	iouMain = 0.1;
	if isfield(cfg, 'eval') && isfield(cfg.eval, 'iouThreshold')
		iouMain = cfg.eval.iouThreshold;
	end

	% Build report.
	report = struct();
	report.meta = struct();
	report.meta.createdAt = char(datetime('now'));
	report.meta.script = mfilename;
	report.meta.matlabVersion = version;
	try
		report.meta.platform = struct('computer', computer, 'arch', computer('arch'));
	catch err
		warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Meta', 'Failed to collect platform info: %s', err.message);
	end
	report.meta.paths = sanitizePathsForExport(paths);
	report.meta.numTemplates = numTemplates;
	report.meta.iouMain = iouMain;
	report.meta.outputDirs = struct();
	try
		repoRoot = fullfile(paths.devRoot, '..');
		report.meta.outputDirs.runOutputDir = sanitizePathForExport(baseOutDir, repoRoot);
		report.meta.outputDirs.evalExportDir = sanitizePathForExport(outDir, repoRoot);
		report.meta.outputDirs.figuresDir = sanitizePathForExport(fullfile(paths.outputRoot, cfg.output.outputDirName), repoRoot);
		report.meta.outputDirs.paperFiguresDir = sanitizePathForExport(fullfile(paths.outputRoot, cfg.paperFigs.outputDirName), repoRoot);
	catch err
		warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Meta', 'Failed to fill report outputDirs: %s', err.message);
	end
	report.meta.notes = {
		'IoU is only used to match boxes when counting TP/FP/FN (not as a tight-localization claim).'
		'This training-free pipeline outputs an explicit per-pixel difference field that we use to rank detections.'
	};

	% Note: this public release does not embed external notes.

	% Store config snapshot (useful for reproducibility).
	report.cfg = cfg;

	% Dataset/run summary.
	labels = string({results.label});
	classes = unique(labels);
	counts = zeros(numel(classes), 1);
	for c = 1:numel(classes)
		counts(c) = sum(labels == classes(c));
	end
	report.run = struct();
	report.run.numImagesRun = numel(results);
	report.run.classes = classes(:);
	report.run.numImagesPerClass = counts;

	% Evaluation summaries.
	if ~isempty(metrics)
		report.eval = metrics;
	else
		report.eval = struct();
	end

	% Curves and detailed tables (recomputed here so exports are consistent).
	report.curves = struct();
	report.curves.prOverall = averagePrecisionFromResults(results, paths, iouMain, false);
	report.curves.prPerClass = struct();
	for c = 1:numel(classes)
		cls = classes(c);
		clsResults = results(labels == cls);
		apC = averagePrecisionFromResults(clsResults, paths, iouMain, false, cls);
		report.curves.prPerClass.(matlab.lang.makeValidName(char(cls))) = apC;
	end
	try
		report.curves.froc = computeFROC(results, paths, iouMain);
	catch err
		warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Curves', 'FROC computation failed: %s', err.message);
		report.curves.froc = struct('thresholds', [], 'fpPerImage', [], 'objRecall', [], 'imgRecall', []);
	end
	try
		report.outcomes = collectDetectionOutcomes(results, paths, iouMain);
	catch err
		warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Curves', 'Outcome collection failed: %s', err.message);
		report.outcomes = struct();
	end

	% Operating point details (aligns with paper figures logic).
	try
		thr = chooseOperatingPointThreshold(report.outcomes, cfg);
		report.operatingPoint = computeOperatingPointCounts(results, paths, iouMain, thr);
		report.operatingPoint.scoreThr = thr;
	catch err
		warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Curves', 'Operating point computation failed: %s', err.message);
	end

	% Cross-validated operating point (threshold selection on calibration folds).
	try
		report.crossValOperatingPoint = computeCrossValidatedOperatingPoint(results, paths, cfg, iouMain);
	catch err
		warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Curves', 'Cross-validated operating point failed: %s', err.message);
		report.crossValOperatingPoint = struct('enabled', false);
	end

	% AP vs IoU sweeps used by paper figures.
	if isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'iouSweep') && ~isempty(cfg.paperFigs.iouSweep)
		report.curves.apVsIoU_custom = computeAPvsIoUSweep(results, classes, paths, cfg, cfg.paperFigs.iouSweep, iouMain);
	end
	if isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'iouSweepCOCO') && ~isempty(cfg.paperFigs.iouSweepCOCO)
		report.curves.apVsIoU_coco = computeAPvsIoUSweep(results, classes, paths, cfg, cfg.paperFigs.iouSweepCOCO, iouMain);
	end

	% Per-image and per-detection tables for later analysis.
	imgTbl = collectPerImageSummary(results, paths, iouMain);
	detTbl = collectPerDetectionTable(results, paths, iouMain);
	report.tables = struct();
	report.tables.imageSummary = imgTbl;
	report.tables.detections = detTbl;

	% Attach paper figure outputs if available.
	if ~isempty(paperOut)
		report.paperFigures = paperOut;
	end

	% ---- Write files ----
	writeMat = true; writeJson = true; writeCsv = true;
	if isfield(cfg, 'export')
		if isfield(cfg.export, 'writeMat'), writeMat = logical(cfg.export.writeMat); end
		if isfield(cfg.export, 'writeJson'), writeJson = logical(cfg.export.writeJson); end
		if isfield(cfg.export, 'writeCsv'), writeCsv = logical(cfg.export.writeCsv); end
	end

	if writeMat
		save(fullfile(outDir, 'evaluation_report.mat'), 'report', '-v7.3');
	end

	if writeCsv
		tryWriteTable(imgTbl, fullfile(outDir, 'image_summary.csv'), warnOnWriteFailures);
		tryWriteTable(detTbl, fullfile(outDir, 'detection_table.csv'), warnOnWriteFailures);
		if ~isempty(metrics) && isfield(metrics, 'summary')
			tryWriteTable(metrics.summary, fullfile(outDir, 'overall_summary.csv'), warnOnWriteFailures);
		end
		if ~isempty(metrics) && isfield(metrics, 'perClassSummary')
			tryWriteTable(metrics.perClassSummary, fullfile(outDir, 'perclass_summary.csv'), warnOnWriteFailures);
		end

		% PR curves
		tryWritePRCurve(report.curves.prOverall, fullfile(outDir, 'pr_overall.csv'), warnOnWriteFailures);
		try
			for c = 1:numel(classes)
				cls = classes(c);
				apC = report.curves.prPerClass.(matlab.lang.makeValidName(char(cls)));
				tryWritePRCurve(apC, fullfile(outDir, sprintf('pr_%s.csv', matlab.lang.makeValidName(char(cls)))), warnOnWriteFailures);
			end
		catch err
			warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Export', 'Failed to export per-class PR curves: %s', err.message);
		end

		% FROC
		try
			froc = report.curves.froc;
			Tf = table(froc.thresholds(:), froc.fpPerImage(:), froc.objRecall(:), froc.imgRecall(:), ...
				'VariableNames', {'scoreThr','fpPerImage','objRecall','imgRecall'});
			tryWriteTable(Tf, fullfile(outDir, 'froc.csv'), warnOnWriteFailures);
		catch err
			warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Export', 'Failed to export FROC: %s', err.message);
		end

		% AP vs IoU sweeps
		tryWriteAPvsIoU(report, outDir, 'apVsIoU_custom', warnOnWriteFailures);
		tryWriteAPvsIoU(report, outDir, 'apVsIoU_coco', warnOnWriteFailures);
	end

	if writeJson
		jsonReport = report;
		% Tables and raw results can be huge in JSON; keep compact.
		if isfield(cfg, 'export') && isfield(cfg.export, 'jsonIncludeRawResults') && ~cfg.export.jsonIncludeRawResults
			if isfield(jsonReport, 'paperFigures')
				% keep (usually small)
			end
			% leave report.tables in place but drop big table contents by default.
			jsonReport = rmfieldIfExists(jsonReport, 'tables');
		end
		if isfield(cfg, 'export') && isfield(cfg.export, 'jsonIncludeDetectionsTable') && cfg.export.jsonIncludeDetectionsTable
			try
				jsonReport.tables = struct('detections', table2struct(detTbl), 'imageSummary', table2struct(imgTbl));
			catch err
				warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Export', 'Failed to embed tables into JSON: %s', err.message);
			end
		end
		writeJsonFile(jsonReport, fullfile(outDir, 'evaluation_report.json'), warnOnWriteFailures);
	end

	% Note: this public release does not write repo-level artifacts.

	% Small README for humans.
	try
		fid = fopen(fullfile(outDir, 'README_eval_export.txt'), 'w');
		if fid > 0
			fprintf(fid, 'Evaluation export folder: %s\n', outDir);
			fprintf(fid, 'Main matching IoU threshold: %.3f\n', iouMain);
			fprintf(fid, '\nKey files:\n');
			fprintf(fid, '  - evaluation_report.mat: full report struct (most complete)\n');
			fprintf(fid, '  - evaluation_report.json: compact summary for quick review\n');
			fprintf(fid, '  - perclass_summary.csv, overall_summary.csv\n');
			fprintf(fid, '  - image_summary.csv: per-image metrics + runtime + template SSIM\n');
			fprintf(fid, '  - detection_table.csv: per-detection TP/FP label + IoU when matched\n');
			fprintf(fid, '  - pr_*.csv, froc.csv, ap_vs_iou_*.csv\n');
			fclose(fid);
		end
	catch err
		warnIfEnabled(warnOnWriteFailures, 'v0_14_eval2:Export', 'Failed to write README: %s', err.message);
	end

	fprintf('[Export] Wrote evaluation bundle to: %s\n', outDir);
end
function sweep = computeAPvsIoUSweep(results, classes, paths, cfg, iouThresholds, iouMain)
	useVOC07 = false;
	if isfield(cfg, 'eval') && isfield(cfg.eval, 'ap') && isfield(cfg.eval.ap, 'useVOC07')
		useVOC07 = cfg.eval.ap.useVOC07;
	end

	sweep = struct();
	sweep.iouThresholds = iouThresholds(:);
	sweep.iouMain = iouMain;

	APoverall = zeros(numel(iouThresholds), 1);
	for i = 1:numel(iouThresholds)
		apOut = averagePrecisionFromResults(results, paths, iouThresholds(i), useVOC07);
		APoverall(i) = apOut.AP;
	end
	sweep.overallAP = APoverall;

	perClass = struct();
	labels = string({results.label});
	for c = 1:numel(classes)
		cls = classes(c);
		clsResults = results(labels == cls);
		APc = zeros(numel(iouThresholds), 1);
		for i = 1:numel(iouThresholds)
			apOut = averagePrecisionFromResults(clsResults, paths, iouThresholds(i), useVOC07, cls);
			APc(i) = apOut.AP;
		end
		perClass.(matlab.lang.makeValidName(char(cls))) = APc;
	end
	sweep.perClassAP = perClass;
end


function imgTbl = collectPerImageSummary(results, paths, iouThr)
	% Per-image evaluation and runtime summary.
	n = numel(results);
	label = strings(n,1);
	testFile = strings(n,1);
	templateFile = strings(n,1);
	templateSSIM = nan(n,1);
	numPred = zeros(n,1);
	meanScore = nan(n,1);
	maxScore = nan(n,1);
	tSel = nan(n,1);
	tLoc = nan(n,1);
	tViz = nan(n,1);
	numGT = nan(n,1);
	TP = nan(n,1);
	FP = nan(n,1);
	FN = nan(n,1);
	poorLoc = nan(n,1);
	hasGT = false(n,1);

	for i = 1:n
		label(i) = string(results(i).label);
		testFile(i) = string(results(i).testFile);
		templateFile(i) = string(results(i).templateFile);
		templateSSIM(i) = results(i).templateSSIM;
		numPred(i) = size(results(i).bboxes, 1);
		if ~isempty(results(i).scores)
			meanScore(i) = mean(results(i).scores(:));
			maxScore(i) = max(results(i).scores(:));
		end
		if isfield(results, 'timeSelectSec'), tSel(i) = results(i).timeSelectSec; end
		if isfield(results, 'timeLocalizeSec'), tLoc(i) = results(i).timeLocalizeSec; end
		if isfield(results, 'timeVisualizeSec'), tViz(i) = results(i).timeVisualizeSec; end

		% Evaluation per image (only when GT exists).
		[~, base, ~] = fileparts(char(results(i).testFile));
		xmlFile = fullfile(paths.annoRoot, char(label(i)), [base '.xml']);
		if ~isfile(xmlFile)
			continue;
		end
		hasGT(i) = true;
		gt = readPascalVocBoxes(xmlFile);
		numGT(i) = size(gt.bboxes, 1);
		[pTp, pFp, pFn, pPoor] = matchDetectionsIoUWithPoorLoc(results(i).bboxes, results(i).scores, gt.bboxes, iouThr);
		TP(i) = pTp; FP(i) = pFp; FN(i) = pFn; poorLoc(i) = pPoor;
	end

	imgTbl = table(label, testFile, templateFile, templateSSIM, hasGT, numGT, numPred, TP, FP, FN, poorLoc, meanScore, maxScore, tSel, tLoc, tViz, ...
		'VariableNames', {'class','testFile','templateFile','templateSSIM','hasGT','numGT','numPred','TP','FP','FN','poorLoc','meanScore','maxScore','timeSelectSec','timeLocalizeSec','timeVisualizeSec'});
end


function detTbl = collectPerDetectionTable(results, paths, iouThr)
	% Per-detection table: assigns each predicted box a TP/FP label + IoU for TP.
	rows = {};
	labels = string({results.label});

	for i = 1:numel(results)
		cls = labels(i);
		[~, base, ~] = fileparts(char(results(i).testFile));
		xmlFile = fullfile(paths.annoRoot, char(cls), [base '.xml']);
		if ~isfile(xmlFile)
			continue;
		end
		gt = readPascalVocBoxes(xmlFile);
		gtB = gt.bboxes;

		predB = results(i).bboxes;
		predS = results(i).scores;
		if isempty(predB)
			continue;
		end
		if isempty(predS)
			predS = ones(size(predB,1),1);
		end
		predS = predS(:);
		if numel(predS) ~= size(predB,1)
			predS = ones(size(predB,1),1);
		end

		[~, order] = sort(predS, 'descend');
		predB = predB(order,:);
		predS = predS(order);

		gtMatched = false(size(gtB,1), 1);
		for d = 1:size(predB,1)
			bestIoU = 0;
			bestJ = NaN;
			if ~isempty(gtB)
				ious = bboxIoU(predB(d,:), gtB);
				[bestIoU, bestJ] = max(ious);
			end
			isTP = ~isempty(gtB) && bestIoU >= iouThr && ~gtMatched(bestJ);
			if isTP
				gtMatched(bestJ) = true;
			end
			isPoorLoc = (~isTP) && (bestIoU > 0);
			rows(end+1, :) = { ...
				char(cls), char(results(i).testFile), d, ...
				predS(d), predB(d,1), predB(d,2), predB(d,3), predB(d,4), ...
				logical(isTP), bestIoU, double(bestJ), logical(isPoorLoc) ...
			};
		end
	end

	if isempty(rows)
		detTbl = table();
		return;
	end

	detTbl = cell2table(rows, 'VariableNames', { ...
		'class','testFile','detRank', ...
		'score','x','y','w','h', ...
		'isTP','bestIoU','bestGTIndex','isPoorLoc' ...
	});
end


function tryWritePRCurve(apOut, outFile, warnOnFailure)
	if nargin < 3, warnOnFailure = false; end
	try
		T = table(apOut.recall(:), apOut.precision(:), 'VariableNames', {'recall','precision'});
		tryWriteTable(T, outFile, warnOnFailure);
	catch err
		warnIfEnabled(warnOnFailure, 'v0_14_eval2:Export', 'Failed to export PR curve: %s', err.message);
	end
end


function tryWriteAPvsIoU(report, outDir, fieldName, warnOnFailure)
	if nargin < 4, warnOnFailure = false; end
	try
		if ~isfield(report, 'curves') || ~isfield(report.curves, fieldName)
			return;
		end
		sweep = report.curves.(fieldName);
		Ti = table(sweep.iouThresholds(:), sweep.overallAP(:), 'VariableNames', {'IoU','AP_overall'});
		tryWriteTable(Ti, fullfile(outDir, sprintf('ap_vs_iou_%s_overall.csv', fieldName)), warnOnFailure);

		% Per-class wide table.
		clsNames = fieldnames(sweep.perClassAP);
		Twide = table(sweep.iouThresholds(:), 'VariableNames', {'IoU'});
		for k = 1:numel(clsNames)
			Twide.(clsNames{k}) = sweep.perClassAP.(clsNames{k})(:);
		end
		tryWriteTable(Twide, fullfile(outDir, sprintf('ap_vs_iou_%s_perclass.csv', fieldName)), warnOnFailure);
	catch err
		warnIfEnabled(warnOnFailure, 'v0_14_eval2:Export', 'Failed to export AP-vs-IoU: %s', err.message);
	end
end


function ok = tryWriteTable(T, outFile, warnOnFailure)
	if nargin < 3, warnOnFailure = false; end
	ok = false;
	try
		[folder,~,~] = fileparts(outFile);
		ensureDir(folder);
		writetable(T, outFile);
		ok = true;
	catch err
		warnIfEnabled(warnOnFailure, 'v0_14_eval2:WriteTable', 'Failed to write %s: %s', outFile, err.message);
	end
end


function ok = writeJsonFile(x, outFile, warnOnFailure)
	if nargin < 3, warnOnFailure = false; end
	ok = false;
	try
		[folder,~,~] = fileparts(outFile);
		ensureDir(folder);
		try
			j = jsonencode(x, 'PrettyPrint', true);
		catch
			j = jsonencode(x);
		end
		fid = fopen(outFile, 'w');
		if fid <= 0
			return;
		end
		fwrite(fid, j);
		fclose(fid);
		ok = true;
	catch err
		warnIfEnabled(warnOnFailure, 'v0_14_eval2:WriteJson', 'Failed to write %s: %s', outFile, err.message);
	end
end


function ensureDir(folder)
	if nargin < 1 || isempty(folder) || isfolder(folder)
		return;
	end
	[ok, msg, msgid] = mkdir(folder);
	if ~ok
		error('v0_14_eval2:mkdirFailed', 'Failed to create folder: %s (%s: %s)', folder, msgid, msg);
	end
end


function warnIfEnabled(enabled, warnId, fmt, varargin)
	if enabled
		warning(warnId, fmt, varargin{:});
	end
end


function s = rmfieldIfExists(s, field)
	if isstruct(s) && isfield(s, field)
		s = rmfield(s, field);
	end
end


function outPaths = sanitizePathsForExport(paths)
	% Avoid leaking machine-specific absolute paths in exported reports.
	outPaths = paths;
	if ~isstruct(outPaths) || ~isfield(outPaths, 'devRoot')
		return;
	end
	repoRoot = fullfile(outPaths.devRoot, '..');
	fn = fieldnames(outPaths);
	for i = 1:numel(fn)
		k = fn{i};
		v = outPaths.(k);
		if ischar(v) || (isstring(v) && isscalar(v))
			outPaths.(k) = sanitizePathForExport(v, repoRoot);
		end
	end
end


function out = sanitizePathForExport(p, repoRoot)
	if isstring(p), p = char(p); end
	if isstring(repoRoot), repoRoot = char(repoRoot); end
	if isempty(p)
		out = p;
		return;
	end

	pN = strrep(p, '\', '/');
	repoN = strrep(repoRoot, '\', '/');

	% Relative paths are fine as-is.
	isAbs = startsWith(pN, '/') || ~isempty(regexp(pN, '^[A-Za-z]:/', 'once'));
	if ~isAbs
		out = p;
		return;
	end

	if ~endsWith(repoN, '/')
		repoN = [repoN '/'];
	end
	if startsWith(pN, repoN)
		out = ['<repo>/' extractAfter(pN, repoN)];
		return;
	end

	homeN = strrep(getenv('HOME'), '\', '/');
	if ~isempty(homeN)
		if ~endsWith(homeN, '/')
			homeN = [homeN '/'];
		end
		if startsWith(pN, homeN)
			out = ['<home>/' extractAfter(pN, homeN)];
			return;
		end
	end

	[~, base, ext] = fileparts(pN);
	out = [base ext];
end


function paths = defaultPaths()
	paths = struct();

	thisFile = mfilename('fullpath');
	devRoot = fileparts(thisFile);

	paths.devRoot = devRoot;
	paths.datasetRoot = pickFirstExistingFolder({
		fullfile(devRoot, 'PCB_DATASET'), ...
		fullfile(devRoot, 'PCB_DATASET(1)'), ...
		fullfile(devRoot, 'PCB_DATASET（1）') ...
	}, fullfile(devRoot, 'PCB_DATASET'));
	paths.defectiveRoot = pickFirstExistingFolder({
		fullfile(paths.datasetRoot, 'images（Defective）'), ...
		fullfile(paths.datasetRoot, 'images(Defective)') ...
	}, fullfile(paths.datasetRoot, 'images（Defective）'));
	paths.gtRoot = pickFirstExistingFolder({
		fullfile(paths.datasetRoot, 'PCB_USED（Defect-free）'), ...
		fullfile(paths.datasetRoot, 'PCB_USED(Defect-free)') ...
	}, fullfile(paths.datasetRoot, 'PCB_USED（Defect-free）'));
	paths.annoRoot = fullfile(paths.datasetRoot, 'Annotations');
	paths.outputRoot = fullfile(devRoot, 'outputs');
end


function folder = pickFirstExistingFolder(candidates, defaultFolder)
	folder = defaultFolder;
	for k = 1:numel(candidates)
		c = candidates{k};
		if isstring(c), c = char(c); end
		if isfolder(c)
			folder = c;
			return;
		end
	end
end


function [defectiveDS, templateFiles] = buildDatastores(paths, cfg)
	% Creates:
	%  - defectiveDS: imageDatastore with Labels from defect class folder name
	%  - templateFiles: cell array of defect-free image filepaths
	if ~isfolder(paths.defectiveRoot)
		error('Defective folder not found: %s', paths.defectiveRoot);
	end
	if ~isfolder(paths.gtRoot)
		error('Defect-free folder not found: %s', paths.gtRoot);
	end

	exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.JPG','.JPEG','.PNG'};
	% Defective images are organized into subfolders per class.
	defectiveDS = imageDatastore(paths.defectiveRoot, ...
		'IncludeSubfolders', true, ...
		'LabelSource', 'foldernames', ...
		'FileExtensions', exts);

	% Templates are all defect-free images (flat folder).
	templateFiles = listFilesWithExts(paths.gtRoot, exts);
	if isempty(templateFiles)
		error('No defect-free template images found in: %s', paths.gtRoot);
	end

	% Deterministic randomness (also controls optional shuffling below)
	rng(cfg.run.randomSeed);

	% Optional shuffle of test-file ordering
	% Enable via: cfg.paths.defectiveShuffle = true;
	if cfg.paths.defectiveShuffle
		n = numel(defectiveDS.Files);
		if n > 1
			order = randperm(n);
			files = defectiveDS.Files(order);
			labels = defectiveDS.Labels(order);

			% Recreate datastore so Files+Labels stay aligned regardless of MATLAB version
			defectiveDS = imageDatastore(files, 'Labels', labels);
		end
	end
end


function results = runPipeline(defectiveDS, templateFiles, paths, cfg)
	% Runs the pipeline per defective image.
	% For each test image: select best template -> localize defects -> visualize.
	outputDir = fullfile(paths.outputRoot, cfg.output.outputDirName);
	if cfg.output.saveFigures
		ensureDir(outputDir);
	end

	numToRun = numel(defectiveDS.Files);
	if ~isempty(cfg.run.maxImages)
		numToRun = min(numToRun, cfg.run.maxImages);
	end

	results = repmat(struct( ...
		'testFile', '', ...
		'label', '', ...
		'templateFile', '', ...
		'templateSSIM', NaN, ...
		'bboxes', zeros(0,4), ...
		'scores', zeros(0,1), ...
		'maskAreaPx', 0, ...
		'timeSelectSec', NaN, ...
		'timeSelectPreSec', NaN, ...
		'timeSelectSSIMSec', NaN, ...
		'timeLocalizeSec', NaN, ...
		'timePreprocessSec', NaN, ...
		'timeResizeSec', NaN, ...
		'timeAlignSec', NaN, ...
		'timeScoreMapSec', NaN, ...
		'timeMaskSec', NaN, ...
		'timeBoxesSec', NaN, ...
		'timeVisualizeSec', NaN, ...
		'figureFile', ''), numToRun, 1);

	templateCache = containers.Map('KeyType','char','ValueType','any');

	% Precompute / load template indices once per run (major speed win).
	[templateIndexSel, templateIndexPre] = prepareTemplateSelectionIndices(templateFiles, paths, cfg);

	for idx = 1:numToRun
		% (A) Load test file + label
		%     - testFile is the defective image
		%     - label is inferred from the parent folder name
		testFile = defectiveDS.Files{idx};
		label = char(defectiveDS.Labels(idx));

		% (B) Template selection by SSIM (best defect-free image)
		%     - uses downscaled grayscale for speed
		%     - ignores border ROI to reduce noise
		tSel = tic;
		[templateFile, templateSSIM, templateTopK, cached, selTiming] = selectBestTemplateBySSIM(testFile, templateFiles, cfg, templateCache, templateIndexSel, templateIndexPre);
		timeSelectSec = toc(tSel);
		if cfg.templateSelection.cache && ~cached
			templateCache(testFile) = struct('templateFile', templateFile, 'templateSSIM', templateSSIM);
		end

		% (C) Defect localization against the selected template
		%     - optional registration (alignment)
		%     - score map + thresholding (within ROI)
		%     - connected components -> bounding boxes
		tLoc = tic;
		[predBboxes, predScores, maskAreaPx, debug, locTiming] = localizeDefects(testFile, templateFile, cfg);
		timeLocalizeSec = toc(tLoc);

		% (D) Visualization (includes padding ROI rectangle)
		%     - cyan dashed rectangle = ROI inner area (border ignored)
		%     - green rectangles = predicted defect regions
		figFile = '';
		timeVisualizeSec = NaN;
		if cfg.output.showFigures || cfg.output.saveFigures
			tViz = tic;
			figFile = visualizeResult(testFile, templateFile, label, templateSSIM, predBboxes, predScores, debug, outputDir, cfg);
			timeVisualizeSec = toc(tViz);
		end

		results(idx).testFile = testFile;
		results(idx).label = label;
		results(idx).templateFile = templateFile;
		results(idx).templateSSIM = templateSSIM;
		results(idx).bboxes = predBboxes;
		results(idx).scores = predScores;
		results(idx).maskAreaPx = maskAreaPx;
		results(idx).timeSelectSec = timeSelectSec;
		results(idx).timeSelectPreSec = selTiming.preselectSec;
		results(idx).timeSelectSSIMSec = selTiming.ssimSec;
		results(idx).timeLocalizeSec = timeLocalizeSec;
		results(idx).timePreprocessSec = locTiming.preprocessSec;
		results(idx).timeResizeSec = locTiming.resizeSec;
		results(idx).timeAlignSec = locTiming.alignSec;
		results(idx).timeScoreMapSec = locTiming.scoreMapSec;
		results(idx).timeMaskSec = locTiming.maskSec;
		results(idx).timeBoxesSec = locTiming.boxesSec;
		results(idx).timeVisualizeSec = timeVisualizeSec;
		results(idx).figureFile = figFile;

		if cfg.debug.verbose
			fprintf(['[%d/%d] %s | label=%s | bestTemplate=%s | bestSSIM=%.4f | ' ...
				'padArea=%.3f padPx=%d | thr=%.3f | maskPx=%d | boxes=%d\n'], ...
				idx, numToRun, stripPath(testFile), label, stripPath(templateFile), templateSSIM, cfg.roi.avoidAreaRatio, debug.padPx, debug.threshold, maskAreaPx, size(predBboxes,1));
			if ~isempty(templateTopK)
				fprintf('        Top-%d templates by SSIM:\n', size(templateTopK,1));
				for t = 1:size(templateTopK,1)
					fprintf('          #%d  ssim=%.4f  %s\n', t, templateTopK.ssim(t), stripPath(templateTopK.file{t}));
				end
			end
		else
			fprintf('[%d/%d] %s | label=%s | template=%s | ssim=%.4f | boxes=%d\n', ...
				idx, numToRun, stripPath(testFile), label, stripPath(templateFile), templateSSIM, size(predBboxes,1));
		end
		fprintf('\n');
	end
end


function paperOut = generatePaperFigures(results, paths, cfg)
	% Generates the full "interesting figures" suite for overall evaluation.
	% All outputs are written into dev/outputs/<cfg.paperFigs.outputDirName>/.
	% This function assumes results were produced by runPipeline().

	paperOut = struct();
	if ~isfield(cfg, 'paperFigs') || ~cfg.paperFigs.enable
		return;
	end

	outDir = fullfile(paths.outputRoot, cfg.paperFigs.outputDirName);
	if ~isfolder(outDir)
		mkdir(outDir);
	end

	supDir = outDir;
	if isfield(cfg.paperFigs, 'presentation') && isfield(cfg.paperFigs.presentation, 'supplementSubdir')
		supDir = fullfile(outDir, cfg.paperFigs.presentation.supplementSubdir);
		if ~isfolder(supDir)
			mkdir(supDir);
		end
	end

	% Choose the IoU threshold used for the main "overall" plots.
	% Note: in this work IoU is only a matching rule for TP/FP/FN counts; it is
	% not meant to imply tight localization.
	iouMain = 0.1;
	if isfield(cfg, 'eval') && isfield(cfg.eval, 'iouThreshold')
		iouMain = cfg.eval.iouThreshold;
	end

	matchNote = sprintf('Matching uses IoU\x2265%.2f (for TP/FP/FN counting; boxes are intentionally coarse)', iouMain);
	showAPInTitles = true;
	showIoUInTitles = true;
	if isfield(cfg.paperFigs, 'presentation')
		if isfield(cfg.paperFigs.presentation, 'showAPInTitles'), showAPInTitles = cfg.paperFigs.presentation.showAPInTitles; end
		if isfield(cfg.paperFigs.presentation, 'showIoUInTitles'), showIoUInTitles = cfg.paperFigs.presentation.showIoUInTitles; end
	end

	classes = unique(string({results.label}));

	% ---- Overall PR curve (micro) + AP ----
	apOverall = averagePrecisionFromResults(results, paths, iouMain, false);
	f = figure('Color','w','Name','Overall PR');
	if isempty(apOverall.recall) || isempty(apOverall.precision)
		plot(0, 0, '.');
		grid on;
		xlabel('Recall'); ylabel('Precision');
		title('Overall PR (matching)');
		addMatchNote(f, sprintf('No PR points available (missing GT or predictions). %s', matchNote), cfg);
	else
		plot(apOverall.recall, apOverall.precision, 'LineWidth', 2);
		grid on;
		xlabel('Recall'); ylabel('Precision');
		if showAPInTitles
			title(sprintf('Overall PR (matching), AP=%.3f', apOverall.AP));
		else
			title('Overall PR (matching)');
		end
		addMatchNote(f, matchNote, cfg);
	end
	exportFigureBundle(f, fullfile(outDir, sprintf('overall_PR_iou%.2f.%s', iouMain, cfg.paperFigs.figureFormat)), cfg, 'paper');
	closeIfHeadless(f, cfg);
	paperOut.apOverall = apOverall;

	% ---- Per-class PR curves ----
	nC = numel(classes);
	combine = true;
	if isfield(cfg.paperFigs, 'perClassPR') && isfield(cfg.paperFigs.perClassPR, 'combineIntoSinglePlot')
		combine = cfg.paperFigs.perClassPR.combineIntoSinglePlot;
	end

	if combine
		% Single plot with legend (one curve per class).
		f = figure('Color','w','Name','Per-class PR (combined)');
		ax = axes(f);
		hold(ax, 'on');
		grid(ax, 'on');

		colors = lines(max(nC, 1));
		lineStyles = {'-','--',':','-.'};
		legendNames = {};
		legendHandles = gobjects(0,1);

		% Collect precision stats for y-limits in 'top'/'auto' modes.
		allP = [];
		for c = 1:nC
			cls = classes(c);
			clsResults = results(strcmp(string({results.label}), cls));
			apC = averagePrecisionFromResults(clsResults, paths, iouMain, false, cls);
			if isempty(apC.recall) || isempty(apC.precision) || apC.numGT == 0
				continue;
			end
			p = apC.precision(:);
			p = p(isfinite(p));
			allP = [allP; p];
		end

		scaleMode = getPerClassPRScaleMode(cfg);
		if strcmpi(scaleMode, 'auto')
			if ~isempty(allP) && median(allP) > 0.85
				scaleMode = 'top';
			else
				scaleMode = 'full';
			end
		end

		for c = 1:nC
			cls = classes(c);
			clsResults = results(strcmp(string({results.label}), cls));
			apC = averagePrecisionFromResults(clsResults, paths, iouMain, false, cls);
			if isempty(apC.recall) || isempty(apC.precision) || apC.numGT == 0
				continue;
			end
			ls = lineStyles{mod(c-1, numel(lineStyles)) + 1};
			col = colors(c, :);
			switch lower(scaleMode)
				case 'log1mp'
					h = semilogy(ax, apC.recall, max(eps, 1 - apC.precision), 'LineWidth', 1.6, 'LineStyle', ls, 'Color', col);
				otherwise
					h = plot(ax, apC.recall, apC.precision, 'LineWidth', 1.8, 'LineStyle', ls, 'Color', col);
			end
			legendHandles(end+1,1) = h;
			legendNames{end+1,1} = char(cls);
		end

		xlim(ax, [0 1]);
		xlabel(ax, 'Recall');
		if strcmpi(scaleMode, 'log1mp')
			ylabel(ax, '1 - Precision');
			ylim(ax, [1e-4 1]);
		else
			ylabel(ax, 'Precision');
			% Apply top/full y-limits once, based on all curves.
			if strcmpi(scaleMode, 'top') && ~isempty(allP)
				topMargin = 0.02;
				minSpan = 0.10;
				if isfield(cfg.paperFigs, 'perClassPR')
					if isfield(cfg.paperFigs.perClassPR, 'topMargin'), topMargin = cfg.paperFigs.perClassPR.topMargin; end
					if isfield(cfg.paperFigs.perClassPR, 'minSpan'), minSpan = cfg.paperFigs.perClassPR.minSpan; end
				end
				pMin = min(allP);
				yMin = max(0, pMin - topMargin);
				if (1 - yMin) < minSpan
					yMin = max(0, 1 - minSpan);
				end
				ylim(ax, [yMin 1]);
			else
				ylim(ax, [0 1]);
			end
		end

		if showIoUInTitles
			title(ax, sprintf('Per-class PR curves (%s)', matchNote), 'Interpreter','none');
		else
			title(ax, 'Per-class PR curves (matching)', 'Interpreter','none');
		end
		addMatchNote(f, matchNote, cfg);
		if ~isempty(legendHandles)
			legend(ax, legendHandles, legendNames, 'Interpreter','none', 'Location','best');
			try
				legend(ax, 'boxoff');
			catch
			end
		end
		exportFigureBundle(f, fullfile(outDir, sprintf('perclass_PR_iou%.2f.%s', iouMain, cfg.paperFigs.figureFormat)), cfg, 'paper');
		closeIfHeadless(f, cfg);
	else
		% Fallback: small multiples.
		f = figure('Color','w','Name','Per-class PR');
		cols = max(1, ceil(sqrt(nC)));
		rows = max(1, ceil(nC/cols));
		tl = tiledlayout(f, rows, cols, 'TileSpacing','compact', 'Padding','compact');
		for c = 1:nC
			cls = classes(c);
			clsResults = results(strcmp(string({results.label}), cls));
			apC = averagePrecisionFromResults(clsResults, paths, iouMain, false, cls);
			nexttile;
			if isempty(apC.recall) || isempty(apC.precision) || apC.numGT == 0
				axis off;
				text(0.05, 0.6, sprintf('%s', cls), 'Interpreter','none', 'FontWeight','bold');
				text(0.05, 0.4, sprintf('No GT / no PR points'), 'Interpreter','none');
			else
				applyPRPlotScaling(apC.recall, apC.precision, cfg, true);
				xlim([0 1]);
				xlabel('R');
				if strcmpi(getPerClassPRScaleMode(cfg), 'log1mP')
					ylabel('1 - P');
				else
					ylabel('P');
				end
				if showAPInTitles
					title(sprintf('%s\nAP=%.3f (GT=%d)', cls, apC.AP, apC.numGT), 'Interpreter','none');
				else
					title(sprintf('%s (GT=%d)', cls, apC.numGT), 'Interpreter','none');
				end
			end
		end
		try
			if showIoUInTitles
				title(tl, sprintf('Per-class PR curves (%s)', matchNote));
			else
				title(tl, 'Per-class PR curves (matching)');
			end
		catch
		end
		addMatchNote(f, matchNote, cfg);
		exportFigureBundle(f, fullfile(outDir, sprintf('perclass_PR_iou%.2f.%s', iouMain, cfg.paperFigs.figureFormat)), cfg, 'paper');
		closeIfHeadless(f, cfg);
	end

	% ---- AP vs IoU sweep (two variants) ----
	if isfield(cfg.paperFigs, 'iouSweep') && ~isempty(cfg.paperFigs.iouSweep)
		makeAPvsIoUFigure(results, classes, paths, cfg, cfg.paperFigs.iouSweep, iouMain, fullfile(supDir, sprintf('AP_vs_IoU_custom.%s', cfg.paperFigs.figureFormat)));
	end
	if isfield(cfg.paperFigs, 'iouSweepCOCO') && ~isempty(cfg.paperFigs.iouSweepCOCO)
		makeAPvsIoUFigure(results, classes, paths, cfg, cfg.paperFigs.iouSweepCOCO, iouMain, fullfile(supDir, sprintf('AP_vs_IoU_COCO.%s', cfg.paperFigs.figureFormat)));
	end

	% ---- Per-class AP bars (at iouMain) ----
	clsAP = zeros(nC,1);
	clsGT = zeros(nC,1);
	for c = 1:nC
		cls = classes(c);
		clsResults = results(strcmp(string({results.label}), cls));
		apC = averagePrecisionFromResults(clsResults, paths, iouMain, false, cls);
		clsAP(c) = apC.AP;
		clsGT(c) = apC.numGT;
	end
	f = figure('Color','w','Name','Per-class AP');
	bar(clsAP);
	grid on;
	set(gca, 'XTick', 1:nC, 'XTickLabel', cellstr(classes));
	xtickangle(35);
	ylabel('AP');
	if showAPInTitles
		title(sprintf('Per-class AP (matching), macro-AP=%.3f', mean(clsAP(~isnan(clsAP)))));
	else
		title('Per-class AP (matching)');
	end
	addMatchNote(f, matchNote, cfg);
	exportFigureBundle(f, fullfile(outDir, sprintf('perclass_AP_bars_iou%.2f.%s', iouMain, cfg.paperFigs.figureFormat)), cfg, 'paper');
	closeIfHeadless(f, cfg);

	% ---- Detection outcomes for additional plots (TP/FP scores, IoU distributions, FROC, calibration) ----
	out = collectDetectionOutcomes(results, paths, iouMain);
	paperOut.outcomes = out;

	% ---- TP IoU hist + CDF ----
	if ~isempty(out.tpIoU)
		f = figure('Color','w','Name','TP IoU distribution');
		tiledlayout(f, 1, 2, 'TileSpacing','compact', 'Padding','compact');
		nexttile;
		histogram(out.tpIoU, 20);
		grid on;
		xlabel('IoU (TP only)'); ylabel('Count');
		title('TP IoU histogram');
		nexttile;
		[F,x] = ecdf(out.tpIoU);
		plot(x, F, 'LineWidth', 2);
		grid on;
		xlabel('IoU'); ylabel('CDF');
		title(sprintf('TP IoU CDF (N=%d)', numel(out.tpIoU)));
		addMatchNote(f, 'Supplement: IoU distribution of matched TPs (context only; not a tight-box claim).', cfg);
		exportFigureBundle(f, fullfile(supDir, sprintf('TP_IoU_hist_cdf_iou%.2f.%s', iouMain, cfg.paperFigs.figureFormat)), cfg, 'paper');
		closeIfHeadless(f, cfg);
	end

	% ---- Score distributions: TP vs FP ----
	f = figure('Color','w','Name','Score distributions');
	hold on;
	if ~isempty(out.tpScores)
		histogram(out.tpScores, 'BinMethod','fd', 'FaceAlpha', 0.5, 'DisplayName', sprintf('TP (N=%d)', numel(out.tpScores)));
	end
	if ~isempty(out.fpScores)
		histogram(out.fpScores, 'BinMethod','fd', 'FaceAlpha', 0.5, 'DisplayName', sprintf('FP (N=%d)', numel(out.fpScores)));
	end
	hold off;
	grid on;
	xlabel('Predicted score'); ylabel('Count');
	title('TP vs FP score distributions (matching)');
	addMatchNote(f, matchNote, cfg);
	legend('Location','best');
	exportFigureBundle(f, fullfile(outDir, sprintf('score_hist_TPFP_iou%.2f.%s', iouMain, cfg.paperFigs.figureFormat)), cfg, 'paper');
	closeIfHeadless(f, cfg);

	% ---- Calibration / reliability diagram ----
	if ~isempty(out.allScores)
		f = figure('Color','w','Name','Calibration');
		[nBinsX, conf, acc, counts] = reliabilityDiagram(out.allScores, out.isTP, 10);
		plot(nBinsX, acc, '-o', 'LineWidth', 2, 'DisplayName','Empirical precision');
		hold on;
		plot([0 1], [0 1], '--', 'DisplayName','Ideal');
		hold off;
		grid on;
		xlim([0 1]); ylim([0 1]);
		xlabel('Predicted score (binned)'); ylabel('Empirical precision');
		title('Reliability diagram (supplement)');
		addMatchNote(f, sprintf('Supplement: score calibration under matching IoU=%.2f.', iouMain), cfg);
		legend('Location','best');
		exportFigureBundle(f, fullfile(supDir, sprintf('calibration_reliability_iou%.2f.%s', iouMain, cfg.paperFigs.figureFormat)), cfg, 'paper');
		closeIfHeadless(f, cfg);
		paperOut.calibration = table(nBinsX(:), conf(:), acc(:), counts(:), 'VariableNames', {'binCenter','meanScore','precision','count'});
	end

	% ---- FROC: FP per image vs recall ----
	froc = computeFROC(results, paths, iouMain);
	f = figure('Color','w','Name','FROC');
	plot(froc.fpPerImage, froc.objRecall, 'LineWidth', 2, 'DisplayName','Object-level recall');
	hold on;
	plot(froc.fpPerImage, froc.imgRecall, 'LineWidth', 2, 'DisplayName','Image-level recall');
	hold off;
	grid on;
	xlabel('False positives per image'); ylabel('Recall');
	title('FROC-style curve (matching)');
	addMatchNote(f, matchNote, cfg);
	legend('Location','best');
	exportFigureBundle(f, fullfile(outDir, sprintf('FROC_iou%.2f.%s', iouMain, cfg.paperFigs.figureFormat)), cfg, 'paper');
	closeIfHeadless(f, cfg);
	paperOut.froc = froc;

	% ---- Error breakdown at an operating point ----
	thr = chooseOperatingPointThreshold(out, cfg);
	op = computeOperatingPointCounts(results, paths, iouMain, thr);
	f = figure('Color','w','Name','Error breakdown');
	bar([op.TP, op.FP, op.FN, op.poorLoc]);
	set(gca, 'XTickLabel', {'TP','FP','FN','PoorLoc'});
	ylabel('Count');
	grid on;
	title(sprintf('Error breakdown at score>=%.3f (matching)', thr));
	addMatchNote(f, matchNote, cfg);
	exportFigureBundle(f, fullfile(outDir, sprintf('error_breakdown_thr%.3f_iou%.2f.%s', thr, iouMain, cfg.paperFigs.figureFormat)), cfg, 'paper');
	closeIfHeadless(f, cfg);
	paperOut.operatingPoint = op;
	paperOut.operatingPoint.thr = thr;

	% ---- Runtime breakdown (if timing fields exist) ----
	if isfield(results, 'timeSelectSec') && isfield(results, 'timeLocalizeSec')
		plotRuntimeBreakdown(results, cfg, fullfile(outDir, sprintf('runtime_breakdown.%s', cfg.paperFigs.figureFormat)));
	end

	% ---- Qualitative montages: top FP / FN / TP ----
	try
		plotQualitativeMontages(results, paths, cfg, iouMain, thr, outDir);
	catch err
		fprintf('[Warn] qualitative montages skipped: %s\n', err.message);
	end

	% ---- Step-by-step panel (template/test/maps/mask/boxes) ----
	try
		plotStepByStepExample(results, paths, cfg, iouMain, thr, outDir);
	catch err
		fprintf('[Warn] step-by-step panel skipped: %s\n', err.message);
	end
end


function makeAPvsIoUFigure(results, classes, paths, cfg, iouThresholds, iouMain, outFile)
	useVOC07 = false;
	if isfield(cfg, 'eval') && isfield(cfg.eval, 'ap') && isfield(cfg.eval.ap, 'useVOC07')
		useVOC07 = cfg.eval.ap.useVOC07;
	end
	AP = zeros(numel(iouThresholds), 1);
	for i = 1:numel(iouThresholds)
		apOut = averagePrecisionFromResults(results, paths, iouThresholds(i), useVOC07);
		AP(i) = apOut.AP;
	end
	f = figure('Color','w','Name','AP vs IoU');
	plot(iouThresholds, AP, '-o', 'LineWidth', 2);
	hold on;
	if ~isempty(iouMain) && isfinite(iouMain)
		xline(iouMain, '--');
	end
	hold off;
	grid on;
	xlabel('IoU threshold'); ylabel('AP (overall)');
	title('AP vs IoU (supplement: sensitivity to matching threshold)');
	addMatchNote(f, sprintf('Supplement: sensitivity to matching IoU. Main matching uses IoU=%.2f (TP/FP/FN criterion).', iouMain), cfg);
	exportFigureBundle(f, outFile, cfg, 'paper');
	closeIfHeadless(f, cfg);

	% Also write a simple CSV for convenience.
	try
		T = table(iouThresholds(:), AP(:), 'VariableNames', {'IoU','AP'});
		[folder, base, ~] = fileparts(outFile);
		writetable(T, fullfile(folder, [base '.csv']));
	catch
	end

	% Per-class AP@IoU curve (optional, small multiples to keep readable)
	if numel(classes) <= 12
		combine = true;
		if isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'perClassAPvsIoU') && isfield(cfg.paperFigs.perClassAPvsIoU, 'combineIntoSinglePlot')
			combine = logical(cfg.paperFigs.perClassAPvsIoU.combineIntoSinglePlot);
		end

		f = figure('Color','w','Name','AP vs IoU (per-class)');
		if combine
			cols = lines(numel(classes));
			styles = {'-','--',':','-.'};
			hold on;
			for c = 1:numel(classes)
				cls = classes(c);
				clsResults = results(strcmp(string({results.label}), cls));
				APc = zeros(numel(iouThresholds), 1);
				for i = 1:numel(iouThresholds)
					apOut = averagePrecisionFromResults(clsResults, paths, iouThresholds(i), useVOC07, cls);
					APc(i) = apOut.AP;
				end
				ls = styles{mod(c-1, numel(styles))+1};
				plot(iouThresholds, APc, [ls 'o'], 'LineWidth', 1.8, 'Color', cols(c,:), 'DisplayName', char(cls));
			end
			if ~isempty(iouMain) && isfinite(iouMain)
				xline(iouMain, '--', 'HandleVisibility','off');
			end
			hold off;
			grid on;
			xlim([min(iouThresholds) max(iouThresholds)]);
			ylim([0 1]);
			xlabel('IoU'); ylabel('AP');
			title('AP vs IoU (per-class)');
			legend('Location','best', 'Interpreter','none');
		else
			cols = max(1, ceil(sqrt(numel(classes))));
			rows = max(1, ceil(numel(classes)/cols));
			tl = tiledlayout(f, rows, cols, 'TileSpacing','compact', 'Padding','compact');
			for c = 1:numel(classes)
				cls = classes(c);
				clsResults = results(strcmp(string({results.label}), cls));
				APc = zeros(numel(iouThresholds), 1);
				for i = 1:numel(iouThresholds)
					apOut = averagePrecisionFromResults(clsResults, paths, iouThresholds(i), useVOC07, cls);
					APc(i) = apOut.AP;
				end
				nexttile;
				plot(iouThresholds, APc, '-o', 'LineWidth', 1.5);
				grid on;
				xlim([min(iouThresholds) max(iouThresholds)]);
				ylim([0 1]);
				title(cls, 'Interpreter','none');
				xlabel('IoU'); ylabel('AP');
			end
			try
				title(tl, 'AP vs IoU (per-class)');
			catch
			end
		end
		addMatchNote(f, sprintf('Supplement: per-class sensitivity to matching IoU. Main matching uses IoU=%.2f.', iouMain), cfg);
		[folder, base, ext] = fileparts(outFile);
		exportFigureBundle(f, fullfile(folder, [base '_perclass' ext]), cfg, 'paper');
		closeIfHeadless(f, cfg);
	end
end


function addMatchNote(fig, note, cfg)
	% Adds a small, consistent note on figures so reviewers don't over-interpret IoU.
	show = false;
	if nargin >= 3 && isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'presentation') && isfield(cfg.paperFigs.presentation, 'showMatchNote')
		show = logical(cfg.paperFigs.presentation.showMatchNote);
	end
	if ~show
		return;
	end
	try
		fontSize = 9;
		col = [0.25 0.25 0.25];
		pos = [0.01 0.01 0.98 0.055];
		if nargin >= 3 && isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'presentation') && isfield(cfg.paperFigs.presentation, 'lowProminence') && cfg.paperFigs.presentation.lowProminence
			fontSize = 7;
			col = [0.35 0.35 0.35];
			pos = [0.01 0.005 0.98 0.045];
		end
		annotation(fig, 'textbox', pos, 'String', note, ...
			'Interpreter','none', 'EdgeColor','none', 'FontSize', fontSize, 'Color', col);
	catch
	end
end


function out = collectDetectionOutcomes(results, paths, iouThr)
	% Returns per-detection outcomes (TP/FP labels + IoU for TPs) at a fixed IoU threshold.
	% Matches greedily per-image, with detections processed in descending score order.

	out = struct();
	out.allScores = zeros(0,1);
	out.isTP = false(0,1);
	out.tpIoU = zeros(0,1);
	out.tpScores = zeros(0,1);
	out.fpScores = zeros(0,1);
	out.numGT = 0;
	out.numImages = 0;

	for i = 1:numel(results)
		cls = string(results(i).label);
		[~, base, ~] = fileparts(results(i).testFile);
		xmlFile = fullfile(paths.annoRoot, char(cls), [base '.xml']);
		if ~isfile(xmlFile)
			continue;
		end
		out.numImages = out.numImages + 1;
		gt = readPascalVocBoxes(xmlFile);
		gtB = gt.bboxes;
		out.numGT = out.numGT + size(gtB,1);

		predB = results(i).bboxes;
		predS = results(i).scores;
		if isempty(predB)
			continue;
		end
		if isempty(predS)
			predS = ones(size(predB,1),1);
		end
		predS = predS(:);
		if numel(predS) ~= size(predB,1)
			predS = ones(size(predB,1),1);
		end
		[~, order] = sort(predS, 'descend');
		predB = predB(order,:);
		predS = predS(order);

		gtMatched = false(size(gtB,1), 1);
		for d = 1:size(predB,1)
			if isempty(gtB)
				out.allScores(end+1,1) = predS(d);
				out.isTP(end+1,1) = false;
				out.fpScores(end+1,1) = predS(d);
				continue;
			end
			ious = bboxIoU(predB(d,:), gtB);
			[bestIoU, j] = max(ious);
			isTP = bestIoU >= iouThr && ~gtMatched(j);
			out.allScores(end+1,1) = predS(d);
			out.isTP(end+1,1) = isTP;
			if isTP
				gtMatched(j) = true;
				out.tpScores(end+1,1) = predS(d);
				out.tpIoU(end+1,1) = bestIoU;
			else
				out.fpScores(end+1,1) = predS(d);
			end
		end
	end
end


function thr = chooseOperatingPointThreshold(outcomes, cfg)
	% Chooses a confidence threshold for "single operating point" plots.
	thr = 0;
	if ~isfield(cfg, 'paperFigs') || ~isfield(cfg.paperFigs, 'operatingPoint')
		return;
	end
	op = cfg.paperFigs.operatingPoint;
	if isnumeric(op) && isscalar(op)
		thr = op;
		return;
	end
	if ischar(op) || isstring(op)
		if strcmpi(string(op), "bestF1")
			thr = bestF1Threshold(outcomes.allScores, outcomes.isTP, outcomes.numGT);
			return;
		end
	end
end


function thr = bestF1Threshold(scores, isTP, numGT)
	% Finds a threshold on score that maximizes global F1.
	thr = 0;
	if isempty(scores) || numGT <= 0
		return;
	end
	[s, order] = sort(scores(:), 'descend');
	isTP = isTP(order);
	tpp = cumsum(isTP);
	fpp = cumsum(~isTP);
	prec = tpp ./ max(eps, tpp + fpp);
	rec = tpp ./ max(eps, numGT);
	f1 = (2 .* prec .* rec) ./ max(eps, prec + rec);
	[~, idx] = max(f1);
	thr = s(idx);
end


function op = computeOperatingPointCounts(results, paths, iouThr, scoreThr)
	% Computes TP/FP/FN counts at a given score threshold.
	% Also computes "poor localization" = predictions that overlap a GT (IoU>0)
	% but are below iouThr or match already-taken GT.
	op = struct('TP',0,'FP',0,'FN',0,'poorLoc',0,'images',0,'numGT',0);
	for i = 1:numel(results)
		cls = string(results(i).label);
		[~, base, ~] = fileparts(results(i).testFile);
		xmlFile = fullfile(paths.annoRoot, char(cls), [base '.xml']);
		if ~isfile(xmlFile)
			continue;
		end
		op.images = op.images + 1;
		gt = readPascalVocBoxes(xmlFile);
		gtB = gt.bboxes;
		op.numGT = op.numGT + size(gtB,1);

		predB = results(i).bboxes;
		predS = results(i).scores;
		if isempty(predB)
			op.FN = op.FN + size(gtB,1);
			continue;
		end
		if isempty(predS)
			predS = ones(size(predB,1),1);
		end
		predS = predS(:);
		keep = predS >= scoreThr;
		predB = predB(keep,:);
		predS = predS(keep);
		if isempty(predB)
			op.FN = op.FN + size(gtB,1);
			continue;
		end
		[tp, fp, fn, poor] = matchDetectionsIoUWithPoorLoc(predB, predS, gtB, iouThr);
		op.TP = op.TP + tp;
		op.FP = op.FP + fp;
		op.FN = op.FN + fn;
		op.poorLoc = op.poorLoc + poor;
	end
end


function idxEval = getEvaluatedResultIndices(results, paths)
	% Returns indices of results entries that have a corresponding GT XML file.
	idxEval = zeros(0,1);
	for i = 1:numel(results)
		cls = string(results(i).label);
		[~, base, ~] = fileparts(results(i).testFile);
		xmlFile = fullfile(paths.annoRoot, char(cls), [base '.xml']);
		if isfile(xmlFile)
			idxEval(end+1,1) = i; %#ok<AGROW>
		end
	end
end


function cv = computeCrossValidatedOperatingPoint(results, paths, cfg, iouThr)
	% Computes K-fold cross-validated operating-point metrics.
	% Threshold is selected on calibration folds, evaluated on held-out fold.
	cv = struct('enabled', false);
	if ~isfield(cfg, 'eval') || ~isfield(cfg.eval, 'cv') || ~isfield(cfg.eval.cv, 'enable') || ~cfg.eval.cv.enable
		return;
	end
	K = 5;
	if isfield(cfg.eval.cv, 'K') && ~isempty(cfg.eval.cv.K)
		K = round(cfg.eval.cv.K);
	end
	seed = 0;
	if isfield(cfg.eval.cv, 'seed') && ~isempty(cfg.eval.cv.seed)
		seed = cfg.eval.cv.seed;
	end

	idxEval = getEvaluatedResultIndices(results, paths);
	n = numel(idxEval);
	if n < 2
		cv.enabled = false;
		cv.reason = 'Not enough evaluated images with GT to run CV.';
		cv.numImages = n;
		return;
	end
	K = max(2, min(K, n));

	rng(seed);
	perm = randperm(n);
	foldId = zeros(n,1);
	for j = 1:n
		foldId(perm(j)) = mod(j-1, K) + 1;
	end

	thr = nan(K,1);
	P = nan(K,1);
	R = nan(K,1);
	F1 = nan(K,1);
	FPPI = nan(K,1);
	imagesPerFold = zeros(K,1);
	gtPerFold = zeros(K,1);

	for f = 1:K
		testMask = (foldId == f);
		calibMask = ~testMask;

		idxCalib = idxEval(calibMask);
		idxTest = idxEval(testMask);
		imagesPerFold(f) = numel(idxTest);

		outCalib = collectDetectionOutcomes(results(idxCalib), paths, iouThr);
		thr(f) = chooseOperatingPointThreshold(outCalib, cfg);

		op = computeOperatingPointCounts(results(idxTest), paths, iouThr, thr(f));
		gtPerFold(f) = op.numGT;
		P(f) = safeDiv(op.TP, op.TP + op.FP);
		R(f) = safeDiv(op.TP, op.TP + op.FN);
		F1(f) = safeDiv(2 * P(f) * R(f), P(f) + R(f));
		FPPI(f) = op.FP / max(1, op.images);
	end

	cv.enabled = true;
	cv.K = K;
	cv.seed = seed;
	cv.iouThreshold = iouThr;
	cv.numImages = n;
	cv.thresholds = thr;
	cv.precision = P;
	cv.recall = R;
	cv.f1 = F1;
	cv.fpPerImage = FPPI;
	cv.imagesPerFold = imagesPerFold;
	cv.gtPerFold = gtPerFold;

	cv.summary = struct();
	cv.summary.thresholdMean = mean(thr, 'omitnan');
	cv.summary.thresholdStd = std(thr, 'omitnan');
	cv.summary.precisionMean = mean(P, 'omitnan');
	cv.summary.precisionStd = std(P, 'omitnan');
	cv.summary.recallMean = mean(R, 'omitnan');
	cv.summary.recallStd = std(R, 'omitnan');
	cv.summary.f1Mean = mean(F1, 'omitnan');
	cv.summary.f1Std = std(F1, 'omitnan');
	cv.summary.fpPerImageMean = mean(FPPI, 'omitnan');
	cv.summary.fpPerImageStd = std(FPPI, 'omitnan');
end


function [tp, fp, fn, poorLoc] = matchDetectionsIoUWithPoorLoc(predB, predS, gtB, iouThr)
	poorLoc = 0;
	if isempty(gtB)
		tp = 0;
		fp = size(predB, 1);
		fn = 0;
		return;
	end
	if isempty(predB)
		tp = 0;
		fp = 0;
		fn = size(gtB, 1);
		return;
	end
	if isempty(predS)
		predS = ones(size(predB,1),1);
	end
	[~, order] = sort(predS, 'descend');
	predB = predB(order,:);
	gtMatched = false(size(gtB, 1), 1);
	tp = 0; fp = 0;
	for i = 1:size(predB,1)
		ious = bboxIoU(predB(i,:), gtB);
		[bestIoU, j] = max(ious);
		if bestIoU >= iouThr && ~gtMatched(j)
			tp = tp + 1;
			gtMatched(j) = true;
		else
			fp = fp + 1;
			if bestIoU > 0
				poorLoc = poorLoc + 1;
			end
		end
	end
	fn = sum(~gtMatched);
end


function froc = computeFROC(results, paths, iouThr)
	% Computes a simple FROC-style sweep over score thresholds.
	% Here we report:
	%  - fpPerImage: average false positives per evaluated image
	%  - objRecall : fraction of GT objects that get matched (TP/numGT)
	%  - imgRecall : fraction of GT images where we detect at least one object

	% Gather all scores so thresholds are data-driven.
	out = collectDetectionOutcomes(results, paths, iouThr);
	if isempty(out.allScores)
		froc = struct('thresholds', [], 'fpPerImage', [], 'objRecall', [], 'imgRecall', []);
		return;
	end

	% Use a manageable number of thresholds (quantiles of the score distribution).
	q = linspace(0, 1, 50);
	thr = unique(quantileSafe(out.allScores, q));
	thr = sort(thr, 'descend');

	fpPerImg = zeros(numel(thr), 1);
	objRec = zeros(numel(thr), 1);
	imgRec = zeros(numel(thr), 1);

	for t = 1:numel(thr)
		op = computeOperatingPointCounts(results, paths, iouThr, thr(t));
		fpPerImg(t) = op.FP / max(1, op.images);
		objRec(t) = op.TP / max(eps, op.numGT);
		% Image-level recall requires counting images with any TP.
		[imgWithGT, imgWithAnyTP] = imageLevelRecallAtThreshold(results, paths, iouThr, thr(t));
		imgRec(t) = imgWithAnyTP / max(1, imgWithGT);
	end

	froc = struct();
	froc.thresholds = thr;
	froc.fpPerImage = fpPerImg;
	froc.objRecall = objRec;
	froc.imgRecall = imgRec;
end


function [imgWithGT, imgWithAnyTP] = imageLevelRecallAtThreshold(results, paths, iouThr, scoreThr)
	imgWithGT = 0;
	imgWithAnyTP = 0;
	for i = 1:numel(results)
		cls = string(results(i).label);
		[~, base, ~] = fileparts(results(i).testFile);
		xmlFile = fullfile(paths.annoRoot, char(cls), [base '.xml']);
		if ~isfile(xmlFile)
			continue;
		end
		gt = readPascalVocBoxes(xmlFile);
		if isempty(gt.bboxes)
			continue;
		end
		imgWithGT = imgWithGT + 1;
		predB = results(i).bboxes;
		predS = results(i).scores;
		if isempty(predB)
			continue;
		end
		if isempty(predS)
			predS = ones(size(predB,1),1);
		end
		keep = predS(:) >= scoreThr;
		predB = predB(keep,:);
		predS = predS(keep);
		if isempty(predB)
			continue;
		end
		[tp, ~, ~] = matchDetectionsIoU(predB, predS, gt.bboxes, iouThr);
		if tp > 0
			imgWithAnyTP = imgWithAnyTP + 1;
		end
	end
end


function [binCenter, meanScore, precision, count] = reliabilityDiagram(scores, isTP, nBins)
	if nargin < 3
		nBins = 10;
	end
	scores = scores(:);
	isTP = logical(isTP(:));

	edges = linspace(0, 1, nBins+1);
	binCenter = (edges(1:end-1) + edges(2:end)) / 2;
	meanScore = nan(nBins, 1);
	precision = nan(nBins, 1);
	count = zeros(nBins, 1);
	for b = 1:nBins
		in = scores >= edges(b) & scores < edges(b+1);
		if b == nBins
			in = scores >= edges(b) & scores <= edges(b+1);
		end
		count(b) = sum(in);
		if count(b) == 0
			continue;
		end
		meanScore(b) = mean(scores(in));
		precision(b) = mean(double(isTP(in)));
	end
	valid = count > 0;
	binCenter = binCenter(valid);
	meanScore = meanScore(valid);
	precision = precision(valid);
	count = count(valid);
end


function plotRuntimeBreakdown(results, cfg, outFile)
	% Runtime breakdown from per-image timing instrumentation.
	sel = [results.timeSelectSec];
	loc = [results.timeLocalizeSec];
	viz = [results.timeVisualizeSec];
	sel = sel(isfinite(sel));
	loc = loc(isfinite(loc));
	viz = viz(isfinite(viz));
	if isempty(sel) && isempty(loc) && isempty(viz)
		return;
	end
	means = [mean(sel), mean(loc), mean(viz)];
	meds = [median(sel), median(loc), median(viz)];
	f = figure('Color','w','Name','Runtime breakdown');
	bar([means; meds]');
	grid on;
	set(gca, 'XTickLabel', {'TemplateSel','Localize','Visualize'});
	legend({'Mean','Median'}, 'Location','best');
	ylabel('Seconds');
	title('Runtime breakdown per image');
	exportFigureBundle(f, outFile, cfg, 'paper');
	closeIfHeadless(f, cfg);
end


function plotQualitativeMontages(results, paths, cfg, iouThr, scoreThr, outDir)
	% Builds three montages: top FP, top FN, top TP.
	maxFP = 9; maxFN = 9; maxTP = 9;
	if isfield(cfg, 'paperFigs')
		if isfield(cfg.paperFigs, 'maxFPExamples'), maxFP = cfg.paperFigs.maxFPExamples; end
		if isfield(cfg.paperFigs, 'maxFNExamples'), maxFN = cfg.paperFigs.maxFNExamples; end
		if isfield(cfg.paperFigs, 'maxTPExamples'), maxTP = cfg.paperFigs.maxTPExamples; end
	end

	% Collect per-image summaries at the operating point.
	summ = collectImageSummaries(results, paths, iouThr, scoreThr);
	classes = unique(string({results.label}));
	labels = string({results.label});

	splitByClass = false;
	if isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'montage') && isfield(cfg.paperFigs.montage, 'splitByClass')
		splitByClass = cfg.paperFigs.montage.splitByClass;
	end

	if splitByClass
		% Write one montage per class (for FP/FN/TP). Uses per-class top-K.
		for c = 1:numel(classes)
			cls = classes(c);
			idxC = find(labels == cls);
			if isempty(idxC)
				continue;
			end
			clsToken = safeToken(cls);

			% FP: rank by FP desc, then maxScore desc; require FP>0.
			candFP = idxC(summ.FP(idxC) > 0);
			if ~isempty(candFP)
				[~, ordFP] = sortrows([summ.FP(candFP), summ.maxScore(candFP)], [-1 -2]);
				idxFPc = candFP(ordFP(1:min(maxFP, numel(ordFP))));
				plotImageMontage(results, paths, cfg, summ, idxFPc, scoreThr, sprintf('%s | Top false positives', cls), outDir, ...
					sprintf('montage_topFP__%s_thr%.3f_iou%.2f', clsToken, scoreThr, iouThr));
			end

			% FN: has GT and zero TP; rank by FN desc.
			candFN = idxC(summ.hasGT(idxC) & summ.TP(idxC) == 0);
			if ~isempty(candFN)
				[~, ordFN] = sort(summ.FN(candFN), 'descend');
				idxFNc = candFN(ordFN(1:min(maxFN, numel(ordFN))));
				plotImageMontage(results, paths, cfg, summ, idxFNc, scoreThr, sprintf('%s | False negatives', cls), outDir, ...
					sprintf('montage_FN__%s_thr%.3f_iou%.2f', clsToken, scoreThr, iouThr));
			end

			% TP: require TP>0; rank by bestIoU desc then maxScore desc.
			candTP = idxC(summ.TP(idxC) > 0);
			if ~isempty(candTP)
				[~, ordTP] = sortrows([summ.bestIoU(candTP), summ.maxScore(candTP)], [-1 -2]);
				idxTPc = candTP(ordTP(1:min(maxTP, numel(ordTP))));
				plotImageMontage(results, paths, cfg, summ, idxTPc, scoreThr, sprintf('%s | Best true positives', cls), outDir, ...
					sprintf('montage_bestTP__%s_thr%.3f_iou%.2f', clsToken, scoreThr, iouThr));
			end
		end
	else
		% Mixed montages (balanced across classes).
		idxFP = selectTopExamplesPerClass(results, summ, classes, 'worstFP', maxFP);
		plotImageMontage(results, paths, cfg, summ, idxFP, scoreThr, 'Top false positives', outDir, sprintf('montage_topFP_thr%.3f_iou%.2f', scoreThr, iouThr));

		fnMask = summ.hasGT & (summ.TP == 0);
		idxFN = find(fnMask);
		idxFN = idxFN(1:min(maxFN, numel(idxFN)));
		plotImageMontage(results, paths, cfg, summ, idxFN, scoreThr, 'False negatives (missed defects)', outDir, sprintf('montage_FN_thr%.3f_iou%.2f', scoreThr, iouThr));

		idxTP = selectTopExamplesPerClass(results, summ, classes, 'bestTP', maxTP);
		plotImageMontage(results, paths, cfg, summ, idxTP, scoreThr, 'Best true positives', outDir, sprintf('montage_bestTP_thr%.3f_iou%.2f', scoreThr, iouThr));
	end
end


function s = safeToken(x)
	% Converts a label into a filename-safe token.
	try
		s = char(string(x));
	catch
		s = char(x);
	end
	s = regexprep(s, '[^a-zA-Z0-9_\-]+', '_');
	if isempty(s)
		s = 'x';
	end
end


function indices = selectTopExamplesPerClass(results, summ, classes, mode, maxTotal)
	% Selects qualitative examples balanced across classes.
	% mode:
	%  - 'worstFP': rank by FP desc, then maxScore desc
	%  - 'bestTP' : rank by bestIoU desc (requires TP>0), then maxScore desc
	if nargin < 5
		maxTotal = 9;
	end
	if isempty(classes)
		indices = zeros(0,1);
		return;
	end

	labels = string({results.label});
	nC = numel(classes);
	perClassK = max(1, ceil(maxTotal / max(1, nC)));
	indices = zeros(0,1);

	for c = 1:nC
		cls = classes(c);
		idxC = find(labels == cls);
		if isempty(idxC)
			continue;
		end
		switch lower(string(mode))
			case "worstfp"
				valid = summ.FP(idxC) > 0;
				key = [summ.FP(idxC), summ.maxScore(idxC)];
				ord = sortrows([(1:numel(idxC))' key], [ -2 -3 ]);
				ordLocal = ord(:,1);
			case "besttp"
				valid = summ.TP(idxC) > 0;
				key = [summ.bestIoU(idxC), summ.maxScore(idxC)];
				ord = sortrows([(1:numel(idxC))' key], [ -2 -3 ]);
				ordLocal = ord(:,1);
			otherwise
				valid = true(size(idxC));
				ordLocal = (1:numel(idxC))';
		end

		idxC = idxC(valid);
		if isempty(idxC)
			continue;
		end
		ordLocal = ordLocal(valid(ordLocal));
		if isempty(ordLocal)
			ordLocal = (1:numel(idxC))';
		end
		idxPick = idxC(ordLocal(1:min(perClassK, numel(ordLocal))));
		indices = [indices; idxPick(:)];
	end

	indices = unique(indices, 'stable');
	if isempty(indices)
		return;
	end

	% If we exceeded maxTotal, re-rank globally using the same key.
	if numel(indices) > maxTotal
		switch lower(string(mode))
			case "worstfp"
				K = [summ.FP(indices), summ.maxScore(indices)];
				ord = sortrows([(1:numel(indices))' K], [ -2 -3 ]);
				indices = indices(ord(1:maxTotal,1));
			case "besttp"
				K = [summ.bestIoU(indices), summ.maxScore(indices)];
				ord = sortrows([(1:numel(indices))' K], [ -2 -3 ]);
				indices = indices(ord(1:maxTotal,1));
			otherwise
				indices = indices(1:maxTotal);
		end
	end
end


function summ = collectImageSummaries(results, paths, iouThr, scoreThr)
	N = numel(results);
	summ = struct();
	summ.TP = zeros(N,1);
	summ.FP = zeros(N,1);
	summ.FN = zeros(N,1);
	summ.poorLoc = zeros(N,1);
	summ.hasGT = false(N,1);
	summ.maxScore = zeros(N,1);
	summ.bestIoU = zeros(N,1);

	for i = 1:N
		cls = string(results(i).label);
		[~, base, ~] = fileparts(results(i).testFile);
		xmlFile = fullfile(paths.annoRoot, char(cls), [base '.xml']);
		if ~isfile(xmlFile)
			continue;
		end
		gt = readPascalVocBoxes(xmlFile);
		gtB = gt.bboxes;
		summ.hasGT(i) = ~isempty(gtB);
		predB = results(i).bboxes;
		predS = results(i).scores;
		if isempty(predB)
			summ.FN(i) = size(gtB,1);
			continue;
		end
		if isempty(predS)
			predS = ones(size(predB,1),1);
		end
		predS = predS(:);
		keep = predS >= scoreThr;
		predB = predB(keep,:);
		predS = predS(keep);
		if isempty(predB)
			summ.FN(i) = size(gtB,1);
			continue;
		end
		summ.maxScore(i) = max(predS);
		[tp, fp, fn, poor] = matchDetectionsIoUWithPoorLoc(predB, predS, gtB, iouThr);
		summ.TP(i) = tp;
		summ.FP(i) = fp;
		summ.FN(i) = fn;
		summ.poorLoc(i) = poor;
		% Best IoU among predictions vs any GT (for ranking best TP examples)
		bestIoU = 0;
		if ~isempty(gtB)
			for d = 1:size(predB,1)
				bestIoU = max(bestIoU, max(bboxIoU(predB(d,:), gtB)));
			end
		end
		summ.bestIoU(i) = bestIoU;
	end
end


function plotImageMontage(results, paths, cfg, summ, indices, scoreThr, figTitle, outDir, baseName)
	if isempty(indices)
		return;
	end
	n = numel(indices);
	cols = max(1, ceil(sqrt(n)));
	rows = max(1, ceil(n / cols));
	f = figure('Color','w','Name',figTitle);
	tl = tiledlayout(f, rows, cols, 'TileSpacing','compact', 'Padding','compact');
	for k = 1:n
		i = indices(k);
		nexttile;
		img = imread(results(i).testFile);
		imshow(img);
		hold on;
		% Draw GT in red.
		cls = string(results(i).label);
		[~, base, ~] = fileparts(results(i).testFile);
		xmlFile = fullfile(paths.annoRoot, char(cls), [base '.xml']);
		if isfile(xmlFile)
			gt = readPascalVocBoxes(xmlFile);
			for g = 1:size(gt.bboxes,1)
				rectangle('Position', gt.bboxes(g,:), 'EdgeColor','r', 'LineWidth', 2);
			end
		end
		% Draw predictions (filtered) in green.
		predB = results(i).bboxes;
		predS = results(i).scores;
		if ~isempty(predB)
			if isempty(predS)
				predS = ones(size(predB,1),1);
			end
			predS = predS(:);
			keep = predS >= scoreThr;
			predB = predB(keep,:);
			predS = predS(keep);
			for b = 1:size(predB,1)
				rectangle('Position', predB(b,:), 'EdgeColor','g', 'LineWidth', 2);
				text(predB(b,1), max(1, predB(b,2)-10), sprintf('%.2f', predS(b)), 'Color','y', 'FontWeight','bold');
			end
		end
		hold off;
		title(sprintf('TP=%d FP=%d FN=%d\n%s', summ.TP(i), summ.FP(i), summ.FN(i), stripPath(results(i).testFile)), 'Interpreter','none', 'FontSize', 8);
	end
	try
		title(tl, figTitle);
	catch
	end
	exportFigureBundle(f, fullfile(outDir, sprintf('%s.%s', baseName, cfg.paperFigs.figureFormat)), cfg, 'paper');
	closeIfHeadless(f, cfg);
end


function plotStepByStepExample(results, paths, cfg, iouThr, scoreThr, outDir)
	% Creates one detailed multi-panel figure for a single image.
	idx = 1;
	if isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'stepByStep')
		if isfield(cfg.paperFigs.stepByStep, 'forceIndex') && ~isempty(cfg.paperFigs.stepByStep.forceIndex)
			idx = cfg.paperFigs.stepByStep.forceIndex;
		else
			mode = 'first';
			if isfield(cfg.paperFigs.stepByStep, 'exampleMode')
				mode = char(cfg.paperFigs.stepByStep.exampleMode);
			end
			summ = collectImageSummaries(results, paths, iouThr, scoreThr);
			classes = unique(string({results.label}));
			switch lower(mode)
				case 'worstfp'
					% Pick the worst FP among the per-class worst-FP champions.
					cands = selectTopExamplesPerClass(results, summ, classes, 'worstFP', numel(classes));
					if isempty(cands)
						[~, idx] = max(summ.FP);
					else
						[~, j] = max(summ.FP(cands));
						idx = cands(j);
					end
				case 'besttp'
					% Pick the best TP among the per-class best-TP champions.
					cands = selectTopExamplesPerClass(results, summ, classes, 'bestTP', numel(classes));
					if isempty(cands)
						[~, idx] = max(summ.bestIoU);
					else
						[~, j] = max(summ.bestIoU(cands));
						idx = cands(j);
					end
				otherwise
					idx = find(summ.hasGT, 1, 'first');
					if isempty(idx), idx = 1; end
			end
		end
	end
	idx = max(1, min(numel(results), idx));

	testFile = results(idx).testFile;
	templateFile = results(idx).templateFile;
	% Re-run localization to obtain debug maps.
	try
		[bb, sc, ~, debug] = localizeDefects(testFile, templateFile, cfg);
	catch
		bb = zeros(0,4);
		sc = zeros(0,1);
		debug = struct();
	end
	if ~isfield(debug, 'templateRGB') || isempty(debug.templateRGB)
		debug.templateRGB = imread(templateFile);
	end
	if ~isfield(debug, 'testRGB') || isempty(debug.testRGB)
		debug.testRGB = imread(testFile);
	end
	if ~isfield(debug, 'ssimMap'), debug.ssimMap = []; end
	if ~isfield(debug, 'absDiffMap'), debug.absDiffMap = []; end
	if ~isfield(debug, 'diffScoreMap'), debug.diffScoreMap = []; end
	if ~isfield(debug, 'defectMask'), debug.defectMask = []; end
	if ~isfield(debug, 'threshold'), debug.threshold = NaN; end

	f = figure('Color','w','Name','Step-by-step');
	tl = tiledlayout(f, 2, 4, 'TileSpacing','compact', 'Padding','compact');
	nexttile; imshow(debug.templateRGB); title('Template');
	nexttile; imshow(debug.testRGB); title('Test');
	nexttile;
	if ~isempty(debug.ssimMap)
		imshow(1 - debug.ssimMap, []);
		title('1-SSIM map');
	else
		axis off; text(0.05, 0.5, 'SSIM map unavailable', 'Interpreter','none');
	end
	nexttile;
	if ~isempty(debug.absDiffMap)
		imshow(debug.absDiffMap, []);
		title('Abs diff');
	else
		axis off; text(0.05, 0.5, 'Abs diff disabled/unavailable', 'Interpreter','none');
	end
	nexttile;
	if ~isempty(debug.diffScoreMap)
		imshow(debug.diffScoreMap, []);
		title('Fused score');
	else
		axis off; text(0.05, 0.5, 'Score map unavailable', 'Interpreter','none');
	end
	nexttile;
	if ~isempty(debug.defectMask)
		imshow(debug.defectMask);
		if isfinite(debug.threshold)
			title(sprintf('Mask (thr=%.3f)', debug.threshold));
		else
			title('Mask');
		end
	else
		axis off; text(0.05, 0.5, 'Mask unavailable', 'Interpreter','none');
	end
	nexttile; imshow(debug.testRGB); hold on;
	for i = 1:size(bb,1)
		rectangle('Position', bb(i,:), 'EdgeColor','g', 'LineWidth', 2);
		if ~isempty(sc)
			text(bb(i,1), max(1, bb(i,2)-10), sprintf('%.2f', sc(i)), 'Color','y', 'FontWeight','bold');
		end
	end
	hold off; title('Detections');
	nexttile; axis off;
	text(0.05, 0.8, sprintf('File: %s\nLabel: %s\nTemplate SSIM: %.4f\nIoU thr: %.2f\nScore thr: %.3f', ...
		stripPath(testFile), string(results(idx).label), results(idx).templateSSIM, iouThr, scoreThr), 'Interpreter','none');
	try
		title(tl, 'Step-by-step example');
	catch
	end
	exportFigureBundle(f, fullfile(outDir, sprintf('step_by_step_example.%s', cfg.paperFigs.figureFormat)), cfg, 'paper');
	closeIfHeadless(f, cfg);
end


function closeIfHeadless(fig, cfg)
	% If running in non-interactive mode, close figures to avoid resource leaks.
	if isfield(cfg, 'output') && isfield(cfg.output, 'showFigures') && ~cfg.output.showFigures
		try
			close(fig);
		catch
		end
	end
end


function [bestTemplateFile, bestSSIM, topK, cached, timing] = selectBestTemplateBySSIM(testFile, templateFiles, cfg, templateCache, templateIndexSel, templateIndexPre)
	% Selects the best defect-free template for a given test image.
	% Uses ROI-masked SSIM; optionally preselects candidates with a cheap metric.
	cached = false;
	topK = table(string.empty(0,1), zeros(0,1), 'VariableNames', {'file','ssim'});
	timing = struct('preselectSec', 0, 'ssimSec', 0, 'cached', false);

	if cfg.templateSelection.cache && isKey(templateCache, testFile)
		entry = templateCache(testFile);
		bestTemplateFile = entry.templateFile;
		bestSSIM = entry.templateSSIM;
		cached = true;
		timing.cached = true;
		return;
	end

	% (1) Preprocess test image at template-selection resolution.
	maxDim = getTemplateSelectionMaxDim(cfg);
	testGraySmall = preprocessForMatching(imread(testFile), maxDim);
	[roiMaskSmall, roiRectSmall] = buildAvoidanceRoi(size(testGraySmall), cfg);
	if isfield(cfg.templateSelection, 'useROICropForSSIM') && cfg.templateSelection.useROICropForSSIM
		testMasked = cropToRoiRect(testGraySmall, roiRectSmall);
	else
		testMasked = testGraySmall;
		testMasked(~roiMaskSmall) = 0;
	end

	% (2) Optional preselect: shortlist candidates quickly.
	idxCandidates = 1:numel(templateFiles);
	if isfield(cfg.templateSelection, 'preselect') && isfield(cfg.templateSelection.preselect, 'enable') && cfg.templateSelection.preselect.enable
		tPre = tic;
		maxDimPre = cfg.templateSelection.preselect.maxDim;
		if isempty(maxDimPre) || ~isfinite(maxDimPre) || maxDimPre <= 0
			maxDimPre = 128;
		end
		maxDimPre = min(maxDimPre, maxDim);
		testPre = preprocessForMatching(imread(testFile), maxDimPre);
		[roiMaskPre, roiRectPre] = buildAvoidanceRoi(size(testPre), cfg);
		if isfield(cfg.templateSelection, 'useROICropForSSIM') && cfg.templateSelection.useROICropForSSIM
			testPreMasked = cropToRoiRect(testPre, roiRectPre);
		else
			testPreMasked = testPre;
			testPreMasked(~roiMaskPre) = 0;
		end

		preScores = -Inf(numel(templateFiles), 1);
		usePar = canUseParfor(cfg);
		if usePar
			parfor k = 1:numel(templateFiles)
				tmp = getTemplateFromIndexOrDisk(templateIndexPre, templateFiles, k, maxDimPre);
				if ~isequal(size(tmp), size(testPre))
					tmp = imresize(tmp, size(testPre));
				end
				if isfield(cfg.templateSelection, 'useROICropForSSIM') && cfg.templateSelection.useROICropForSSIM
					tmpMasked = cropToRoiRect(tmp, roiRectPre);
				else
					tmpMasked = tmp;
					tmpMasked(~roiMaskPre) = 0;
				end
				preScores(k) = computePreselectScore(testPreMasked, tmpMasked, cfg);
			end
		else
			for k = 1:numel(templateFiles)
				tmp = getTemplateFromIndexOrDisk(templateIndexPre, templateFiles, k, maxDimPre);
				if ~isequal(size(tmp), size(testPre))
					tmp = imresize(tmp, size(testPre));
				end
				if isfield(cfg.templateSelection, 'useROICropForSSIM') && cfg.templateSelection.useROICropForSSIM
					tmpMasked = cropToRoiRect(tmp, roiRectPre);
				else
					tmpMasked = tmp;
					tmpMasked(~roiMaskPre) = 0;
				end
				preScores(k) = computePreselectScore(testPreMasked, tmpMasked, cfg);
			end
		end

		topK = cfg.templateSelection.preselect.topK;
		if isempty(topK) || ~isfinite(topK) || topK <= 0
			topK = min(3, numel(templateFiles));
		end
		[~, orderPre] = sort(preScores, 'descend');
		idxCandidates = orderPre(1:min(numel(orderPre), round(topK)));
		timing.preselectSec = toc(tPre);
	end

	% (3) SSIM on shortlisted candidates.
	scores = NaN(numel(templateFiles), 1);
	tSS = tic;
	usePar = canUseParfor(cfg);
	if usePar
		scoresCand = NaN(numel(idxCandidates), 1);
		parfor ii = 1:numel(idxCandidates)
			k = idxCandidates(ii);
			tmp = getTemplateFromIndexOrDisk(templateIndexSel, templateFiles, k, maxDim);
			if ~isequal(size(tmp), size(testGraySmall))
				tmp = imresize(tmp, size(testGraySmall));
			end
			if isfield(cfg.templateSelection, 'useROICropForSSIM') && cfg.templateSelection.useROICropForSSIM
				tmpMasked = cropToRoiRect(tmp, roiRectSmall);
			else
				tmpMasked = tmp;
				tmpMasked(~roiMaskSmall) = 0;
			end
				scoresCand(ii) = computeSSIMScalarSafe(testMasked, tmpMasked, cfg);
		end
		scores(idxCandidates) = scoresCand;
	else
		for ii = 1:numel(idxCandidates)
			k = idxCandidates(ii);
			tmp = getTemplateFromIndexOrDisk(templateIndexSel, templateFiles, k, maxDim);
			if ~isequal(size(tmp), size(testGraySmall))
				tmp = imresize(tmp, size(testGraySmall));
			end
			if isfield(cfg.templateSelection, 'useROICropForSSIM') && cfg.templateSelection.useROICropForSSIM
				tmpMasked = cropToRoiRect(tmp, roiRectSmall);
			else
				tmpMasked = tmp;
				tmpMasked(~roiMaskSmall) = 0;
			end
			scores(k) = computeSSIMScalarSafe(testMasked, tmpMasked, cfg);
		end
	end
	timing.ssimSec = toc(tSS);

	[bestSSIM, bestIdx] = max(scores);
	if isempty(bestIdx) || isnan(bestSSIM)
		bestIdx = idxCandidates(1);
		bestSSIM = scores(bestIdx);
	end
	bestTemplateFile = templateFiles{bestIdx};

	if cfg.templateSelection.reportTopK > 0
		% Build a debug table listing the top-K templates.
		valid = ~isnan(scores);
		idxValid = find(valid);
		K = min(cfg.templateSelection.reportTopK, numel(idxValid));
		if K <= 0
			topK = table(string.empty(0,1), zeros(0,1), 'VariableNames', {'file','ssim'});
		else
			[sortedScores, orderLocal] = sort(scores(idxValid), 'descend');
			order = idxValid(orderLocal);
			topFiles = templateFiles(order(1:K));
			topK = table(topFiles(:), sortedScores(1:K), 'VariableNames', {'file','ssim'});
		end
	end
end


function imgGray = preprocessForMatching(img, maxDim)
	% Common preprocessing used for template matching.
	% - convert to grayscale
	% - normalize to [0,1]
	% - optional downscale (maxDim)
	imgGray = im2graySafe(img);
	imgGray = im2single(imgGray);

	% Mild normalization (avoid aggressive CLAHE that can alter defects)
	imgGray = rescaleSafe(imgGray);

	if ~isempty(maxDim) && isfinite(maxDim)
		sz = size(imgGray);
		scale = maxDim / max(sz(1), sz(2));
		if scale < 1
			imgGray = imresize(imgGray, scale);
		end
	end
end


function [templateIndexSel, templateIndexPre] = prepareTemplateSelectionIndices(templateFiles, paths, cfg)
	% Prepares (or loads) preprocessed template images for template selection.
	% Returns indices for SSIM resolution and preselect resolution.
	maxDim = getTemplateSelectionMaxDim(cfg);
	templateIndexSel = loadOrBuildTemplateIndex(templateFiles, maxDim, paths, cfg);

	templateIndexPre = struct('maxDim',[],'files',{templateFiles},'images',[]);
	if isfield(cfg, 'templateSelection') && isfield(cfg.templateSelection, 'preselect') && isfield(cfg.templateSelection.preselect, 'enable') && cfg.templateSelection.preselect.enable
		maxDimPre = cfg.templateSelection.preselect.maxDim;
		if isempty(maxDimPre) || ~isfinite(maxDimPre) || maxDimPre <= 0
			maxDimPre = 128;
		end
		maxDimPre = min(maxDimPre, maxDim);
		templateIndexPre = loadOrBuildTemplateIndex(templateFiles, maxDimPre, paths, cfg);
	end
end


function idx = loadOrBuildTemplateIndex(templateFiles, maxDim, paths, cfg)
	% Loads a template index from disk when enabled+available, otherwise builds it.
	idx = struct('maxDim', maxDim, 'files', {templateFiles}, 'images', []);
	if isempty(maxDim) || ~isfinite(maxDim)
		% Native-sized caching is typically too heavy; keep empty index.
		return;
	end

	useDisk = isfield(cfg, 'templateSelection') && isfield(cfg.templateSelection, 'diskCache') && isfield(cfg.templateSelection.diskCache, 'enable') && cfg.templateSelection.diskCache.enable;
	forceRebuild = false;
	cacheDir = '';
	if useDisk
		if isfield(cfg.templateSelection.diskCache, 'forceRebuild')
			forceRebuild = cfg.templateSelection.diskCache.forceRebuild;
		end
		if isfield(cfg.templateSelection.diskCache, 'dir')
			cacheDir = cfg.templateSelection.diskCache.dir;
		end
		if isempty(cacheDir)
			cacheDir = fullfile(paths.devRoot, 'cache');
		end
		if ~isfolder(cacheDir)
			mkdir(cacheDir);
		end
		cacheFile = fullfile(cacheDir, sprintf('template_index_maxDim_%d.mat', round(maxDim)));
		if ~forceRebuild && isfile(cacheFile)
			S = load(cacheFile, 'idx');
			if isfield(S, 'idx') && isfield(S.idx, 'files') && numel(S.idx.files) == numel(templateFiles)
				try
					if isequal(S.idx.files(:), templateFiles(:)) && isfield(S.idx, 'images') && numel(S.idx.images) == numel(templateFiles)
						idx = S.idx;
						return;
					end
				catch
					% fall through to rebuild
				end
			end
		end
	end

	% Build in-memory index (and optionally save).
	if isfield(cfg, 'templateSelection') && isfield(cfg.templateSelection, 'cachePreprocessedTemplates')
		useMem = cfg.templateSelection.cachePreprocessedTemplates;
	else
		useMem = true;
	end
	if ~useMem
		return;
	end

	imgs = cell(numel(templateFiles), 1);
	for k = 1:numel(templateFiles)
		imgs{k} = preprocessForMatching(imread(templateFiles{k}), maxDim);
	end
	idx.images = imgs;

	if useDisk
		try
			save(cacheFile, 'idx', '-v7');
		catch
			% ignore disk-cache failures
		end
	end
end


function img = getTemplateFromIndexOrDisk(templateIndex, templateFiles, k, maxDim)
	% Fetches preprocessed template image k from index if present, else loads on demand.
	if isstruct(templateIndex) && isfield(templateIndex, 'images') && ~isempty(templateIndex.images)
		img = templateIndex.images{k};
		return;
	end
	% fallback: load+preprocess on demand
	img = preprocessForMatching(imread(templateFiles{k}), maxDim);
	% keep as single to reduce per-call cost
	if ~isa(img, 'single')
		img = im2single(img);
	end
end


function maxDim = getTemplateSelectionMaxDim(cfg)
	% Returns the working long-side max dimension for template selection.
	% Priority:
	%  1) cfg.templateSelection.downscaleMaxDim (if non-empty and finite)
	%  2) cfg.templateSelection.downscalePreset (mapped via getTargetLongSide)
	%  3) otherwise: native (Inf)
	maxDim = Inf;
	if ~isfield(cfg, 'templateSelection')
		return;
	end
	if isfield(cfg.templateSelection, 'downscaleMaxDim')
		v = cfg.templateSelection.downscaleMaxDim;
		if ~isempty(v) && isfinite(v) && v > 0
			maxDim = v;
			return;
		end
	end

	preset = 'native';
	if isfield(cfg.templateSelection, 'downscalePreset')
		preset = cfg.templateSelection.downscalePreset;
	end
	tmp = struct();
	if isfield(cfg, 'resize')
		tmp.resize = cfg.resize;
	else
		tmp.resize = struct();
	end
	% Override preset to template-selection preset (can differ from localization preset).
	tmp.resize.preset = preset;
	maxDim = getTargetLongSide(tmp);
	if isempty(maxDim) || ~isfinite(maxDim) || maxDim <= 0
		maxDim = Inf;
	end
end


function ok = canUseParfor(cfg)
	ok = false;
	if ~isfield(cfg, 'templateSelection') || ~isfield(cfg.templateSelection, 'parallel') || ~isfield(cfg.templateSelection.parallel, 'enable')
		return;
	end
	if ~cfg.templateSelection.parallel.enable
		return;
	end
	try
		ok = license('test', 'Distrib_Computing_Toolbox');
	catch
		ok = false;
	end
end


function score = computePreselectScore(testImg, templateImg, cfg)
	% Cheap similarity score where higher means more similar.
	metric = 'corr';
	if isfield(cfg, 'templateSelection') && isfield(cfg.templateSelection, 'preselect') && isfield(cfg.templateSelection.preselect, 'metric')
		metric = lower(string(cfg.templateSelection.preselect.metric));
	end
	switch metric
		case "mad"
			score = -mean(abs(single(testImg(:)) - single(templateImg(:))));
		otherwise
			score = corr2Safe(testImg, templateImg);
	end
end


function c = corr2Safe(A, B)
	A = single(A);
	B = single(B);
	A = A - mean(A(:));
	B = B - mean(B(:));
	den = sqrt(sum(A(:).^2) * sum(B(:).^2)) + eps;
	c = sum(A(:) .* B(:)) / den;
end


function args = getSSIMNameValueArgs(cfg)
	% Returns SSIM name-value args based on cfg.ssim.
	% NOTE: supported args vary across MATLAB versions; computeSSIM*Safe will fall back.
	args = {};
	if ~isfield(cfg, 'ssim') || ~isfield(cfg.ssim, 'useExplicitParams') || ~cfg.ssim.useExplicitParams
		return;
	end
	if isfield(cfg.ssim, 'DynamicRange') && ~isempty(cfg.ssim.DynamicRange)
		args(end+1:end+2) = {'DynamicRange', cfg.ssim.DynamicRange};
	end
	if isfield(cfg.ssim, 'K1') && ~isempty(cfg.ssim.K1)
		args(end+1:end+2) = {'K1', cfg.ssim.K1};
	end
	if isfield(cfg.ssim, 'K2') && ~isempty(cfg.ssim.K2)
		args(end+1:end+2) = {'K2', cfg.ssim.K2};
	end
	if isfield(cfg.ssim, 'GaussianWeights') && ~isempty(cfg.ssim.GaussianWeights)
		args(end+1:end+2) = {'GaussianWeights', logical(cfg.ssim.GaussianWeights)};
	end
	if isfield(cfg.ssim, 'WindowSize') && ~isempty(cfg.ssim.WindowSize)
		args(end+1:end+2) = {'WindowSize', cfg.ssim.WindowSize};
	end
end


function s = computeSSIMScalarSafe(A, B, cfg)
	% Computes scalar SSIM. Tries explicit params first, then falls back.
	args = getSSIMNameValueArgs(cfg);
	if ~isempty(args)
		try
			s = ssim(A, B, args{:});
			return;
		catch
			% fall through
		end
	end
	try
		s = ssim(A, B);
	catch
		% Fallback if SSIM is unavailable.
		try
			s = corr2Safe(A, B);
		catch
			s = -Inf;
		end
	end
end


function [s, map] = computeSSIMMapSafe(A, B, cfg)
	% Computes SSIM scalar and per-pixel map.
	% If map is unsupported, returns a constant map filled with scalar SSIM.
	args = getSSIMNameValueArgs(cfg);
	if ~isempty(args)
		try
			[s, map] = ssim(A, B, args{:});
			return;
		catch
			% fall through
		end
	end
	try
		[s, map] = ssim(A, B);
	catch
		% Fallback if SSIM map not available
		s = computeSSIMScalarSafe(A, B, cfg);
		map = ones(size(A), 'like', A) * s;
	end
end


function cropImg = cropToRoiRect(img, roiRect)
	% Crops img to roiRect [x y w h] with clamping.
	H = size(img,1); W = size(img,2);
	x1 = max(1, round(roiRect(1)));
	y1 = max(1, round(roiRect(2)));
	w = max(1, round(roiRect(3)));
	h = max(1, round(roiRect(4)));
	x2 = min(W, x1 + w - 1);
	y2 = min(H, y1 + h - 1);
	cropImg = img(y1:y2, x1:x2);
end


function [predBboxes, predScores, maskAreaPx, debug, timing] = localizeDefects(testFile, templateFile, cfg)
	% Core defect localization.
	% High-level: preprocess -> (optional) resize -> (optional) register -> score map -> mask -> boxes.
	timing = struct('preprocessSec', 0, 'resizeSec', 0, 'alignSec', 0, 'scoreMapSec', 0, 'maskSec', 0, 'boxesSec', 0);
	testRGB0 = imread(testFile);
	templateRGB0 = imread(templateFile);

	% Preprocess (grayscale + mild normalization).
	tPre = tic;
	testGray0 = preprocessForDiff(testRGB0);
	templateGray0 = preprocessForDiff(templateRGB0);
	timing.preprocessSec = toc(tPre);

	% Optional downscale for speed (registration + per-pixel maps are expensive).
	procScale = 1.0;
	if isfield(cfg, 'resize') && isfield(cfg.resize, 'enable') && cfg.resize.enable
		targetLongSide = getTargetLongSide(cfg);
		pairLongSide = max([size(testGray0,1), size(testGray0,2), size(templateGray0,1), size(templateGray0,2)]);
		if isfinite(targetLongSide) && targetLongSide >= pairLongSide
			% Selected preset is higher than the native resolution; use native instead.
			procScale = 1.0;
			if isfield(cfg, 'debug') && isfield(cfg.debug, 'verbose') && cfg.debug.verbose
				fprintf('  [Info] Resize preset "%s" (targetLongSide=%d) >= native long side (%d); using native resolution.\n', ...
					string(cfg.resize.preset), round(targetLongSide), round(pairLongSide));
			end
		else
			[procScale, ~] = computeScaleForPair(size(testGray0), size(templateGray0), targetLongSide);
			if procScale < 1 && isfield(cfg, 'debug') && isfield(cfg.debug, 'verbose') && cfg.debug.verbose
				fprintf('  [Info] Resizing for speed: preset="%s", targetLongSide=%d, scale=%.4f.\n', ...
					string(cfg.resize.preset), round(targetLongSide), procScale);
			end
		end
	end

	localCfg = cfg;
	tResize = tic;
	if procScale < 1
		testRGB = imresize(testRGB0, procScale);
		templateRGB = imresize(templateRGB0, procScale);
		testGray = imresize(testGray0, procScale);
		templateGray = imresize(templateGray0, procScale);

		if isfield(cfg, 'resize') && isfield(cfg.resize, 'scaleMaskParams') && cfg.resize.scaleMaskParams
			localCfg.mask.minAreaPx = max(1, round(cfg.mask.minAreaPx * procScale * procScale));
			localCfg.mask.closeRadiusPx = max(1, round(cfg.mask.closeRadiusPx * procScale));
			localCfg.mask.dilateRadiusPx = max(1, round(cfg.mask.dilateRadiusPx * procScale));
		end
	else
		testRGB = testRGB0;
		templateRGB = templateRGB0;
		testGray = testGray0;
		templateGray = templateGray0;
	end
	timing.resizeSec = toc(tResize);

	% Registration and per-pixel maps assume the same grid.
	if ~isequal(size(testGray), size(templateGray))
		testGray = imresize(testGray, size(templateGray));
	end

	% Optional registration (align test -> template).
	tAlign = tic;
	[testAlignedGray, tform] = alignToTemplate(testGray, templateGray, localCfg);
	timing.alignSec = toc(tAlign);

	% Score map (higher => more different).
	tMap = tic;
	[diffScoreMap, ssimMap, absDiffMap] = computeDiffScoreMap(testAlignedGray, templateGray, localCfg);
	timing.scoreMapSec = toc(tMap);

	% Threshold within ROI + morphology cleanup.
	tMask = tic;
	[defectMask, thr, roiMask, roiRect, padPx] = scoreMapToMask(diffScoreMap, localCfg);
	timing.maskSec = toc(tMask);
	maskAreaPx = nnz(defectMask);

	% Connected components -> bounding boxes.
	tBoxes = tic;
	[predBboxesProc, predScores] = maskToBboxes(defectMask, diffScoreMap, localCfg);

	% Map boxes back to original coordinates for evaluation and downstream use.
	% If procScale < 1, then x_proc = x_orig * procScale  =>  x_orig = x_proc / procScale
	if procScale < 1
		predBboxes = scaleBboxes(predBboxesProc, 1 / procScale);
	else
		predBboxes = predBboxesProc;
	end
	timing.boxesSec = toc(tBoxes);

	% Debug payload (for visualization)
	debug = struct();
	debug.testRGB = testRGB;
	debug.templateRGB = templateRGB;
	debug.testGray = testGray;
	debug.templateGray = templateGray;
	debug.testAlignedGray = testAlignedGray;
	debug.ssimMap = ssimMap;
	debug.absDiffMap = absDiffMap;
	debug.diffScoreMap = diffScoreMap;
	debug.defectMask = defectMask;
	debug.threshold = thr;
	debug.tform = tform;

	debug.roiMask = roiMask;
	debug.roiRect = roiRect;
	debug.padPx = padPx;
	debug.procScale = procScale;
	debug.originalSize = size(testGray0);
	debug.procSize = size(testGray);
	debug.bboxesProc = predBboxesProc;
	if isfield(cfg, 'resize') && isfield(cfg.resize, 'visualizeOnOriginal') && cfg.resize.visualizeOnOriginal
		debug.testRGBOriginal = testRGB0;
		debug.templateRGBOriginal = templateRGB0;
	end
	debug.localCfg = localCfg;
end


function enabled = isTimingEnabled(cfg)
	enabled = isfield(cfg, 'debug') && isfield(cfg.debug, 'timing') && isfield(cfg.debug.timing, 'enable') && cfg.debug.timing.enable;
end


function t = timingStart(cfg, label)
	% Starts a timing section. Returns a tic() handle (numeric).
	t = tic;
	if isTimingEnabled(cfg)
		fprintf('[%s] [Timing] START %s\n', nowTimestamp(), label);
	end
end


function elapsed = timingEnd(cfg, tStart, label)
	% Ends a timing section and prints elapsed time.
	elapsed = toc(tStart);
	if isTimingEnabled(cfg)
		minPrint = 0;
		if isfield(cfg.debug.timing, 'minPrintSec')
			minPrint = cfg.debug.timing.minPrintSec;
		end
		if elapsed >= minPrint
			fprintf('[%s] [Timing] END   %s | %.3fs\n', nowTimestamp(), label, elapsed);
		end
	end
end


function ts = nowTimestamp()
	% Timestamp string for log lines.
	try
		ts = char(datetime('now','Format','HH:mm:ss.SSS'));
	catch
		ts = datestr(now, 'HH:MM:SS.FFF');
	end
end


function imgGray = preprocessForDiff(img)
	% Preprocessing for differencing (kept mild to avoid removing defects).
	imgGray = im2graySafe(img);
	imgGray = im2single(imgGray);
	imgGray = rescaleSafe(imgGray);
end


function [diffScoreMap, ssimMap, absDiffMap] = computeDiffScoreMap(testAlignedGray, templateGray, cfg)
	% Builds the per-pixel maps used for localization.
	%
	% Outputs
	%  - ssimMap: per-pixel SSIM from ssim(test, template). Higher means more similar.
	%  - absDiffMap: per-pixel |test-template| intensity difference, rescaled to [0,1].
	%  - diffScoreMap: combined "difference" score that we later threshold.
	%
	% Combination (conceptually)
	%   raw = useSSIM*(1-ssimMap) + useAbs*(w*absDiffMap),  where w = cfg.diff.absDiffWeight
	%   diffScoreMap = rescale(raw)  (optionally using ROI-only rescaling)
	%
	% Fallback
	%  - If the SSIM function cannot return a per-pixel map, we fall back to a
	%    constant map filled with the scalar SSIM score.
	ssimMap = [];
	absDiffMap = [];
	[roiMask, ~, ~] = buildAvoidanceRoi(size(testAlignedGray), cfg);

	diffScoreMap = zeros(size(testAlignedGray), 'like', testAlignedGray);

	if cfg.diff.useSSIMMap
		try
			[~, ssimMap] = computeSSIMMapSafe(testAlignedGray, templateGray, cfg);
		catch
			% Fallback if SSIM map not available
			ssimMap = ones(size(testAlignedGray), 'like', testAlignedGray) * computeSSIMScalarSafe(testAlignedGray, templateGray, cfg);
		end
		diffScoreMap = diffScoreMap + (1 - ssimMap);
	end

	if cfg.diff.useAbsDiff
		absDiffMap = imabsdiff(testAlignedGray, templateGray);
		if isfield(cfg, 'diff') && isfield(cfg.diff, 'rescaleUsingROI') && cfg.diff.rescaleUsingROI
			absDiffMap = rescaleSafeWithinMask(absDiffMap, roiMask);
		else
			absDiffMap = rescaleSafe(absDiffMap);
		end
		diffScoreMap = diffScoreMap + cfg.diff.absDiffWeight * absDiffMap;
	end

	if isfield(cfg, 'diff') && isfield(cfg.diff, 'rescaleUsingROI') && cfg.diff.rescaleUsingROI
		diffScoreMap = rescaleSafeWithinMask(diffScoreMap, roiMask);
	else
		diffScoreMap = rescaleSafe(diffScoreMap);
	end
end


function [testAlignedGray, tform] = alignToTemplate(testGray, templateGray, cfg)
	% Aligns (registers) the test image to the template.
	% Why: per-pixel differencing is very sensitive to small shifts/rotations.
	% How: uses imregcorr() to estimate a transform, then imwarp() onto the
	% template grid. If registration fails or is rejected by validation, this
	% function simply returns the original test image (no alignment).
	testAlignedGray = testGray;
	tform = [];

	if ~cfg.registration.enable
		if isfield(cfg, 'debug') && isfield(cfg.debug, 'verbose') && cfg.debug.verbose
			fprintf('  [Info] Registration disabled; skipping alignment step.\n');
		end
		return;
	end

	primaryType = cfg.registration.type;
	tryTypes = {primaryType};
	if isfield(cfg.registration, 'fallbackToTranslation') && cfg.registration.fallbackToTranslation
		if ~strcmpi(primaryType, 'translation')
			tryTypes{end+1} = 'translation';
		end
	end

	% Similarity before alignment (for validation)
	beforeSSIM = NaN;
	if isfield(cfg.registration, 'validateTransform') && cfg.registration.validateTransform
		[beforeSSIM, ~] = computeValidationSSIM(testGray, templateGray, cfg);
	end

	lastErr = [];
	for k = 1:numel(tryTypes)
		regType = tryTypes{k};
		try
			candidateTform = imregcorr(testGray, templateGray, regType);
			Rfixed = imref2d(size(templateGray));
			candidateAligned = imwarp(testGray, candidateTform, 'OutputView', Rfixed);

			accept = true;
			afterSSIM = NaN;
			if isfield(cfg.registration, 'validateTransform') && cfg.registration.validateTransform
				[afterSSIM, ~] = computeValidationSSIM(candidateAligned, templateGray, cfg);
				minGain = 0.0;
				if isfield(cfg.registration, 'minSSIMGain')
					minGain = cfg.registration.minSSIMGain;
				end
				if ~(afterSSIM >= beforeSSIM + minGain)
					accept = false;
					if isfield(cfg, 'debug') && isfield(cfg.debug, 'verbose') && cfg.debug.verbose
						fprintf('  [Warn] Registration rejected (%s): SSIM %.4f -> %.4f (minGain=%.4f).\n', regType, beforeSSIM, afterSSIM, minGain);
					end
				end
			end

			if accept
				tform = candidateTform;
				testAlignedGray = candidateAligned;
				if isfield(cfg, 'debug') && isfield(cfg.debug, 'verbose') && cfg.debug.verbose
					if isfield(cfg.registration, 'validateTransform') && cfg.registration.validateTransform
						fprintf('  [Info] Registration accepted (%s): SSIM %.4f -> %.4f.\n', regType, beforeSSIM, afterSSIM);
					else
						fprintf('  [Info] Registration succeeded (%s).\n', regType);
					end
				end
				return;
			end
		catch err
			lastErr = err;
			if isfield(cfg, 'debug') && isfield(cfg.debug, 'verbose') && cfg.debug.verbose
				fprintf('  [Warn] Registration failed (%s). Using fallback if available.\n', regType);
				fprintf('         %s\n', err.message);
			end
		end
	end

	% If registration fails (or gets rejected), proceed without it.
	if isfield(cfg, 'debug') && isfield(cfg.debug, 'verbose') && cfg.debug.verbose
		if ~isempty(lastErr)
			fprintf('  [Warn] Registration unavailable; proceeding without alignment.\n');
		else
			fprintf('  [Info] Registration produced no acceptable transform; proceeding without alignment.\n');
		end
	end
	tform = [];
	testAlignedGray = testGray;
end


function [mask, thr, roiMask, roiRect, padPx] = scoreMapToMask(diffScoreMap, cfg)
	% Turns a per-pixel difference score into a clean binary mask.
	% Steps: ignore border via ROI -> threshold by a high quantile (ROI only) ->
	% morphology cleanup -> ready for connected-components + boxes.
	%
	% Key inputs from cfg: roi.avoidAreaRatio, mask.quantile, mask.minAreaPx,
	% mask.closeRadiusPx, mask.dilateRadiusPx.
	%
	% Outputs: mask (logical), thr (chosen score threshold), plus ROI info for plots.
	[roiMask, roiRect, padPx] = buildAvoidanceRoi(size(diffScoreMap), cfg);

	% Threshold is computed inside the ROI only.
	roiScores = diffScoreMap(roiMask);
	thr = quantileSafe(roiScores(:), cfg.mask.quantile);

	mask = (diffScoreMap > thr) & roiMask;

	mask = bwareaopen(mask, cfg.mask.minAreaPx) & roiMask;
	mask = imclose(mask, strel('disk', cfg.mask.closeRadiusPx)) & roiMask;
	mask = imdilate(mask, strel('disk', cfg.mask.dilateRadiusPx)) & roiMask;
end



function [bboxes, scores] = maskToBboxes(mask, scoreMap, cfg)
	% Converts binary mask -> bounding boxes.
	% score is a robust statistic inside each connected component (configurable).
	% Recommended default: a high quantile (e.g., 0.95) to reduce sensitivity to
	% single hot pixels while still rewarding strong, concentrated responses.
	CC = bwconncomp(mask);
	stats = regionprops(CC, scoreMap, 'BoundingBox', 'PixelIdxList');
	if isempty(stats)
		bboxes = zeros(0,4);
		scores = zeros(0,1);
		return;
	end

	bboxes = vertcat(stats.BoundingBox);
	scores = zeros(numel(stats), 1);

	method = 'quantile';
	q = 0.95;
	if nargin >= 3 && isfield(cfg, 'bbox')
		if isfield(cfg.bbox, 'scoreMethod')
			method = cfg.bbox.scoreMethod;
		end
		if isfield(cfg.bbox, 'scoreQuantile')
			q = cfg.bbox.scoreQuantile;
		end
	end

	for i = 1:numel(stats)
		pix = stats(i).PixelIdxList;
		vals = scoreMap(pix);
		switch lower(string(method))
			case "mean"
				scores(i) = mean(vals);
			case "max"
				scores(i) = max(vals);
			otherwise
				% quantile (robust default)
				scores(i) = quantileSafe(vals(:), q);
		end
	end

	[scores, order] = sort(scores, 'descend');
	bboxes = bboxes(order, :);
end


function figFile = visualizeResult(testFile, templateFile, label, templateSSIM, bboxes, scores, debug, outputDir, cfg)
	% Visualizes:
	%  - Template and Test (with ROI rectangle)
	%  - Score map and mask
	%  - Detections (bboxes)
	%  - Optional intermediate maps (SSIM and abs-diff)
	figFile = '';

	procScale = 1.0;
	if isfield(debug, 'procScale')
		procScale = debug.procScale;
	end

	useOriginal = false;
	if isfield(cfg, 'resize') && isfield(cfg.resize, 'visualizeOnOriginal') && cfg.resize.visualizeOnOriginal
		useOriginal = isfield(debug, 'testRGBOriginal') && isfield(debug, 'templateRGBOriginal');
	end

	if useOriginal
		testRGB = debug.testRGBOriginal;
		templateRGB = debug.templateRGBOriginal;
		roiRectDraw = debug.roiRect;
		padPxDraw = debug.padPx;
		if procScale < 1
			roiRectDraw = scaleRect(debug.roiRect, 1 / procScale);
			padPxDraw = round(debug.padPx / procScale);
		end
		bboxesDraw = bboxes;
	else
		testRGB = debug.testRGB;
		templateRGB = debug.templateRGB;
		roiRectDraw = debug.roiRect;
		padPxDraw = debug.padPx;
		bboxesDraw = bboxes;
		if procScale < 1
			bboxesDraw = scaleBboxes(bboxes, procScale);
		end
	end

	f = figure('Visible', ternary(cfg.output.showFigures, 'on', 'off'), 'Color', 'w');
	set(f, 'Name', sprintf('%s | %s', label, stripPath(testFile)), 'NumberTitle', 'off');

	if cfg.debug.showIntermediateMaps
		tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
	else
		tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
	end

	nexttile;
	imshow(templateRGB);
	hold on;
	if ~isempty(roiRectDraw)
		rectangle('Position', roiRectDraw, 'EdgeColor', 'c', 'LineWidth', 2, 'LineStyle', '--');
	end
	hold off;
	title(sprintf('Template (SSIM=%.4f)\n%s', templateSSIM, stripPath(templateFile)), 'Interpreter', 'none');

	nexttile;
	imshow(testRGB);
	hold on;
	if ~isempty(roiRectDraw)
		rectangle('Position', roiRectDraw, 'EdgeColor', 'c', 'LineWidth', 2, 'LineStyle', '--');
	end
	hold off;
	if procScale < 1
		title(sprintf('Test (%s) | scale=%.3f | padPx=%d\n%s', label, procScale, padPxDraw, stripPath(testFile)), 'Interpreter', 'none');
	else
		title(sprintf('Test (%s) | padPx=%d\n%s', label, padPxDraw, stripPath(testFile)), 'Interpreter', 'none');
	end

	nexttile;
	imshow(debug.diffScoreMap, []);
	hold on;
	contour(debug.defectMask, [1 1], 'r', 'LineWidth', 1);
	if ~isempty(debug.roiRect)
		rectangle('Position', debug.roiRect, 'EdgeColor', 'c', 'LineWidth', 2, 'LineStyle', '--');
	end
	hold off;
	title(sprintf('ScoreMap (thr=%.3f) + mask', debug.threshold));

	nexttile;
	imshow(testRGB);
	hold on;
	if ~isempty(roiRectDraw)
		rectangle('Position', roiRectDraw, 'EdgeColor', 'c', 'LineWidth', 2, 'LineStyle', '--');
	end
	for i = 1:size(bboxesDraw, 1)
		bb = bboxesDraw(i, :);
		rectangle('Position', bb, 'EdgeColor', 'g', 'LineWidth', 2);
		if ~isempty(scores)
			text(bb(1), max(1, bb(2)-10), sprintf('%.2f', scores(i)), 'Color', 'y', 'FontSize', 10, 'FontWeight', 'bold');
		end
	end
	hold off;
	title(sprintf('Detections: %d boxes', size(bboxes,1)));

	if cfg.debug.showIntermediateMaps
		nexttile;
		if ~isempty(debug.ssimMap)
			imshow(1 - debug.ssimMap, []);
			title('1 - SSIM map');
		else
			imshow(zeros(size(debug.testAlignedGray), 'like', debug.testAlignedGray));
			title('SSIM map unavailable');
		end

		nexttile;
		if ~isempty(debug.absDiffMap)
			imshow(debug.absDiffMap, []);
			title('Abs diff map');
		else
			imshow(zeros(size(debug.testAlignedGray), 'like', debug.testAlignedGray));
			title('Abs diff disabled');
		end
	end

	drawnow;

	if cfg.output.saveFigures
		[~, base, ~] = fileparts(testFile);
		outName = sprintf('%s__%s.%s', label, base, cfg.output.figureFormat);
		figFile = fullfile(outputDir, outName);
		exportFigureBundle(f, figFile, cfg, 'output');
	end

	if ~cfg.output.showFigures
		close(f);
	end
end


function metrics = evaluateDetections(results, paths, cfg)
% Evaluates predicted boxes against Pascal VOC XML boxes (per image).

	% This evaluation is class-aware based on the defective folder name.
	% For each test image, it loads the corresponding XML from:
	%   Annotations/<class>/<basename>.xml

	classes = unique(string({results.label}));
	perClass = struct();
	TPv = zeros(numel(classes), 1);
	FPv = zeros(numel(classes), 1);
	FNv = zeros(numel(classes), 1);
	Precv = nan(numel(classes), 1);
	Recv = nan(numel(classes), 1);
	F1v = nan(numel(classes), 1);

	% Object-level totals
	overallTP = 0; overallFP = 0; overallFN = 0;
	imagesEvaluated = 0;
	imagesWithGT = 0;
	imagesWithPred = 0;
	imagesWithAnyTP = 0;

	for c = 1:numel(classes)
		cls = classes(c);
		clsResults = results(strcmp(string({results.label}), cls));

		TP = 0; FP = 0; FN = 0;
		clsImagesEvaluated = 0;
		clsImagesWithGT = 0;
		clsImagesWithPred = 0;
		clsImagesWithAnyTP = 0;

		for i = 1:numel(clsResults)
			testFile = clsResults(i).testFile;
			[~, base, ~] = fileparts(testFile);
			xmlFile = fullfile(paths.annoRoot, char(cls), [base '.xml']);
			if ~isfile(xmlFile)
				continue;
			end
			clsImagesEvaluated = clsImagesEvaluated + 1;
			imagesEvaluated = imagesEvaluated + 1;

			gt = readPascalVocBoxes(xmlFile);
			predB = clsResults(i).bboxes;
			predS = clsResults(i).scores;

			iouThr = 0.5;
			if isfield(cfg, 'eval') && isfield(cfg.eval, 'iouThreshold')
				iouThr = cfg.eval.iouThreshold;
			end
			[tp, fp, fn] = matchDetectionsIoU(predB, predS, gt.bboxes, iouThr);
			TP = TP + tp;
			FP = FP + fp;
			FN = FN + fn;

			hasGT = ~isempty(gt.bboxes);
			hasPred = ~isempty(predB) && size(predB, 1) > 0;
			hasAnyTP = tp > 0;
			clsImagesWithGT = clsImagesWithGT + double(hasGT);
			clsImagesWithPred = clsImagesWithPred + double(hasPred);
			clsImagesWithAnyTP = clsImagesWithAnyTP + double(hasAnyTP);
			imagesWithGT = imagesWithGT + double(hasGT);
			imagesWithPred = imagesWithPred + double(hasPred);
			imagesWithAnyTP = imagesWithAnyTP + double(hasAnyTP);
		end

		prec = safeDiv(TP, TP + FP);
		rec = safeDiv(TP, TP + FN);
		f1 = safeDiv(2 * prec * rec, prec + rec);

		TPv(c) = TP;
		FPv(c) = FP;
		FNv(c) = FN;
		Precv(c) = prec;
		Recv(c) = rec;
		F1v(c) = f1;

		imgPrec = safeDiv(clsImagesWithAnyTP, clsImagesWithPred);
		imgRec = safeDiv(clsImagesWithAnyTP, clsImagesWithGT);
		perClass.(matlab.lang.makeValidName(char(cls))) = table(TP, FP, FN, prec, rec, f1, ...
			clsImagesEvaluated, clsImagesWithGT, clsImagesWithPred, clsImagesWithAnyTP, imgPrec, imgRec, ...
			'VariableNames', {'TP','FP','FN','prec','rec','f1','imagesEvaluated','imagesWithGT','imagesWithPred','imagesWithAnyTP','imgPrec','imgRec'});

		overallTP = overallTP + TP;
		overallFP = overallFP + FP;
		overallFN = overallFN + FN;
	end

	overallPrec = safeDiv(overallTP, overallTP + overallFP);
	overallRec = safeDiv(overallTP, overallTP + overallFN);
	overallF1 = safeDiv(2 * overallPrec * overallRec, overallPrec + overallRec);
	overallImgPrec = safeDiv(imagesWithAnyTP, imagesWithPred);
	overallImgRec = safeDiv(imagesWithAnyTP, imagesWithGT);

	metrics = struct();
	metrics.perClass = perClass;
	metrics.perClassSummary = table(classes(:), TPv, FPv, FNv, Precv, Recv, F1v, ...
		'VariableNames', {'class','TP','FP','FN','prec','rec','f1'});
	metrics.summary = table(overallTP, overallFP, overallFN, overallPrec, overallRec, overallF1, ...
		imagesEvaluated, imagesWithGT, imagesWithPred, imagesWithAnyTP, overallImgPrec, overallImgRec, ...
		'VariableNames', {'TP','FP','FN','prec','rec','f1','imagesEvaluated','imagesWithGT','imagesWithPred','imagesWithAnyTP','imgPrec','imgRec'});

	% ---- Additional metric: AP / mAP (PR-curve based) ----
	if isfield(cfg, 'eval') && isfield(cfg.eval, 'ap') && isfield(cfg.eval.ap, 'enable') && cfg.eval.ap.enable
		iouThresholds = [];
		if isfield(cfg.eval.ap, 'iouThresholds')
			iouThresholds = cfg.eval.ap.iouThresholds;
		end
		if isempty(iouThresholds)
			iouThresholds = cfg.eval.iouThreshold;
		end
		useVOC07 = false;
		if isfield(cfg.eval.ap, 'useVOC07')
			useVOC07 = cfg.eval.ap.useVOC07;
		end

		apPerClass = struct();
		apVals = zeros(numel(iouThresholds), 1);
		for t = 1:numel(iouThresholds)
			apOut = averagePrecisionFromResults(results, paths, iouThresholds(t), useVOC07);
			apVals(t) = apOut.AP;
		end
		mAP = mean(apVals(~isnan(apVals)));
		if isempty(mAP) || isnan(mAP)
			mAP = NaN;
		end

		% Optional per-class AP (can be expensive if many classes + many thresholds)
		if isfield(cfg.eval, 'reportPerClass') && cfg.eval.reportPerClass
			classes = unique(string({results.label}));
			for c = 1:numel(classes)
				cls = classes(c);
				clsResults = results(strcmp(string({results.label}), cls));
				clsAPs = zeros(numel(iouThresholds), 1);
				for t = 1:numel(iouThresholds)
					apOut = averagePrecisionFromResults(clsResults, paths, iouThresholds(t), useVOC07, cls);
					clsAPs(t) = apOut.AP;
				end
				apPerClass.(matlab.lang.makeValidName(char(cls))) = table(mean(clsAPs(~isnan(clsAPs))), ...
					clsAPs(:)', ...
					'VariableNames', {'mAP','APs'});
			end
		end

		metrics.ap = struct();
		metrics.ap.iouThresholds = iouThresholds;
		metrics.ap.perClass = apPerClass;
		metrics.ap.summary = table(mAP, apVals(:)', useVOC07, ...
			'VariableNames', {'mAP','APs','useVOC07'});
	end
end


function out = averagePrecisionFromResults(results, paths, iouThr, useVOC07, forcedClass)
	% Computes AP at a specific IoU threshold from the pipeline results.
	% We rank detections by their score (confidence), then mark TP/FP via greedy
	% IoU matching *within each image*.
	if nargin < 5
		forcedClass = "";
	end

	% Collect GT per evaluable image + detection list.
	gtByImg = {};
	gtMatchedByImg = {};
	detScores = zeros(0,1);
	detBboxes = zeros(0,4);
	detImg = zeros(0,1);

	imgCount = 0;
	numGT = 0;
	for i = 1:numel(results)
		cls = string(results(i).label);
		if strlength(string(forcedClass)) > 0
			cls = string(forcedClass);
		end
		[~, base, ~] = fileparts(results(i).testFile);
		xmlFile = fullfile(paths.annoRoot, char(cls), [base '.xml']);
		if ~isfile(xmlFile)
			continue;
		end

		imgCount = imgCount + 1;
		gt = readPascalVocBoxes(xmlFile);
		gtByImg{imgCount} = gt.bboxes;
		gtMatchedByImg{imgCount} = false(size(gt.bboxes, 1), 1);
		numGT = numGT + size(gt.bboxes, 1);

		predB = results(i).bboxes;
		predS = results(i).scores;
		if isempty(predB)
			continue;
		end
		if isempty(predS)
			predS = ones(size(predB,1), 1);
		end
		predS = predS(:);
		if numel(predS) ~= size(predB,1)
			predS = ones(size(predB,1), 1);
		end

		detScores = [detScores; predS];
		detBboxes = [detBboxes; predB];
		detImg = [detImg; repmat(imgCount, size(predB,1), 1)];
	end

	out = struct();
	out.iouThreshold = iouThr;
	out.numGT = numGT;
	out.numDetections = numel(detScores);

	if numGT == 0
		out.AP = NaN;
		out.precision = zeros(0,1);
		out.recall = zeros(0,1);
		return;
	end
	if isempty(detScores)
		out.AP = 0;
		out.precision = zeros(0,1);
		out.recall = zeros(0,1);
		return;
	end

	% Rank detections by confidence.
	[~, order] = sort(detScores, 'descend');
	detScores = detScores(order);
	detBboxes = detBboxes(order, :);
	detImg = detImg(order);

	tp = zeros(numel(detScores), 1);
	fp = zeros(numel(detScores), 1);
	for d = 1:numel(detScores)
		imgId = detImg(d);
		gtB = gtByImg{imgId};
		if isempty(gtB)
			fp(d) = 1;
			continue;
		end
		ious = bboxIoU(detBboxes(d, :), gtB);
		[bestIoU, j] = max(ious);
		if bestIoU >= iouThr && ~gtMatchedByImg{imgId}(j)
			tp(d) = 1;
			gtMatchedByImg{imgId}(j) = true;
		else
			fp(d) = 1;
		end
	end

	cumTP = cumsum(tp);
	cumFP = cumsum(fp);
	prec = cumTP ./ max(eps, (cumTP + cumFP));
	rec = cumTP ./ max(eps, numGT);

out.precision = prec;
	out.recall = rec;

	if useVOC07
		ap = 0;
		for thr = 0:0.1:1
			p = 0;
			idx = find(rec >= thr);
			if ~isempty(idx)
				p = max(prec(idx));
			end
			ap = ap + p / 11;
		end
	else
		mrec = [0; rec; 1];
		mpre = [0; prec; 0];
		for i = numel(mpre)-1:-1:1
			mpre(i) = max(mpre(i), mpre(i+1));
		end
		ap = 0;
		for i = 1:numel(mrec)-1
			ap = ap + (mrec(i+1) - mrec(i)) * mpre(i+1);
		end
	end
	out.AP = ap;
end


function [s, roiMask] = computeValidationSSIM(movingGray, fixedGray, cfg)
	% Computes a scalar similarity score used to validate registration quality.
	% By default we use SSIM on ROI-masked images (to avoid border artifacts).
	roiMask = true(size(fixedGray));
	if isfield(cfg, 'registration') && isfield(cfg.registration, 'validationUseROI') && cfg.registration.validationUseROI
		[roiMask, ~, ~] = buildAvoidanceRoi(size(fixedGray), cfg);
	end
	A = fixedGray;
	B = movingGray;
	A(~roiMask) = 0;
	B(~roiMask) = 0;
	try
		s = computeSSIMScalarSafe(B, A, cfg);
	catch
		% If SSIM is unavailable for any reason, fall back to normalized correlation.
		s = corr2(B(roiMask), A(roiMask));
	end
	if isnan(s) || isinf(s)
		s = -Inf;
	end
end


function y = rescaleSafeWithinMask(x, mask)
	% Rescales values inside mask to [0,1] and sets outside-mask pixels to 0.
	% This prevents extreme border values (often non-informative) from compressing
	% the dynamic range within the ROI.
	if isempty(mask)
		y = rescaleSafe(x);
		return;
	end
	mask = logical(mask);
	y = zeros(size(x), 'like', x);
	vals = x(mask);
	if isempty(vals)
		return;
	end
	vals = single(vals);
	minv = min(vals(:));
	maxv = max(vals(:));
	if maxv <= minv
		y(mask) = 0;
	else
		y(mask) = (x(mask) - minv) ./ max(eps, (maxv - minv));
	end
end


function gt = readPascalVocBoxes(xmlFile)
	% Parses Pascal VOC XML and returns GT bboxes.
	doc = xmlread(xmlFile);

	% Optional image size fields (for dataset statistics).
	imgW = NaN; imgH = NaN; imgD = NaN;
	try
		sizeNodes = doc.getElementsByTagName('size');
		if sizeNodes.getLength > 0
			sz = sizeNodes.item(0);
			wNode = sz.getElementsByTagName('width');
			hNode = sz.getElementsByTagName('height');
			dNode = sz.getElementsByTagName('depth');
			if wNode.getLength > 0, imgW = str2double(char(wNode.item(0).getFirstChild.getData)); end
			if hNode.getLength > 0, imgH = str2double(char(hNode.item(0).getFirstChild.getData)); end
			if dNode.getLength > 0, imgD = str2double(char(dNode.item(0).getFirstChild.getData)); end
		end
	catch
		% leave as NaN
	end
	objects = doc.getElementsByTagName('object');

	bboxes = zeros(objects.getLength, 4);
	names = strings(objects.getLength, 1);

	for i = 0:objects.getLength-1
		obj = objects.item(i);
		nameNode = obj.getElementsByTagName('name').item(0);
		names(i+1) = string(char(nameNode.getFirstChild.getData));

		bnd = obj.getElementsByTagName('bndbox').item(0);
		xmin = str2double(char(bnd.getElementsByTagName('xmin').item(0).getFirstChild.getData));
		ymin = str2double(char(bnd.getElementsByTagName('ymin').item(0).getFirstChild.getData));
		xmax = str2double(char(bnd.getElementsByTagName('xmax').item(0).getFirstChild.getData));
		ymax = str2double(char(bnd.getElementsByTagName('ymax').item(0).getFirstChild.getData));

		w = max(0, xmax - xmin + 1);
		h = max(0, ymax - ymin + 1);
		bboxes(i+1, :) = [xmin, ymin, w, h];
	end

	gt = struct();
	gt.bboxes = bboxes;
	gt.names = names;
	gt.width = imgW;
	gt.height = imgH;
	gt.depth = imgD;
end


function [tp, fp, fn] = matchDetectionsIoU(predB, predS, gtB, iouThr)
	% Greedy matching between predicted boxes and GT boxes using IoU.
	% Each GT can match at most one prediction.
	if isempty(gtB)
		tp = 0;
		fp = size(predB, 1);
		fn = 0;
		return;
	end
	if isempty(predB)
		tp = 0;
		fp = 0;
		fn = size(gtB, 1);
		return;
	end

	% Sort predictions by score (descending)
	if isempty(predS)
		predS = ones(size(predB,1), 1);
	end
	[~, order] = sort(predS, 'descend');
	predB = predB(order, :);

	gtMatched = false(size(gtB, 1), 1);
	tp = 0; fp = 0;

	for i = 1:size(predB, 1)
		ious = bboxIoU(predB(i, :), gtB);
		[bestIoU, j] = max(ious);
		if bestIoU >= iouThr && ~gtMatched(j)
			tp = tp + 1;
			gtMatched(j) = true;
		else
			fp = fp + 1;
		end
	end

	fn = sum(~gtMatched);
end


function ious = bboxIoU(box, boxes)
% box: [x y w h], boxes: Nx4
	x1 = box(1);
	y1 = box(2);
	x2 = box(1) + box(3);
	y2 = box(2) + box(4);

	xx1 = max(x1, boxes(:, 1));
	yy1 = max(y1, boxes(:, 2));
	xx2 = min(x2, boxes(:, 1) + boxes(:, 3));
	yy2 = min(y2, boxes(:, 2) + boxes(:, 4));

	w = max(0, xx2 - xx1);
	h = max(0, yy2 - yy1);
	inter = w .* h;

	areaA = max(0, (x2 - x1)) * max(0, (y2 - y1));
	areaB = max(0, boxes(:, 3) .* boxes(:, 4));

	ious = inter ./ max(eps, areaA + areaB - inter);
end


function out = listFilesWithExts(folder, exts)
	% Utility: list files in a folder matching any extension.
	out = {};
	for i = 1:numel(exts)
		d = dir(fullfile(folder, ['*' exts{i}]));
		d = d(~[d.isdir]);
		for k = 1:numel(d)
			out{end+1,1} = fullfile(folder, d(k).name);
		end
	end

	% On case-insensitive file systems (common on macOS), patterns like '*.jpg'
	% can also match '.JPG', so the same file may be collected multiple times.
	out = unique(out);
end


function g = im2graySafe(img)
	% Utility: grayscale conversion compatible with multiple MATLAB versions.
	if ndims(img) == 2
		g = img;
	else
		% Use im2gray if available (R2020a+), else rgb2gray
		if exist('im2gray', 'file')
			g = im2gray(img);
		else
			g = rgb2gray(img);
		end
	end
end


function s = stripPath(p)
	[~, name, ext] = fileparts(p);
	s = [name ext];
end


function y = safeDiv(a, b)
	if b == 0
		y = 0;
	else
		y = a / b;
	end
end


function out = ternary(cond, a, b)
	if cond
		out = a;
	else
		out = b;
	end
end


function q = quantileSafe(x, qLevel)
% quantile() is not available in some older MATLAB installs.
	if exist('quantile', 'file')
		q = quantile(x, qLevel);
	else
		q = prctile(x, qLevel * 100);
	end
end


function y = rescaleSafe(x)
% rescale() is newer; keep a fallback.
	if exist('rescale', 'file')
		y = rescale(x);
	else
		x = single(x);
		minx = min(x(:));
		maxx = max(x(:));
		y = (x - minx) ./ max(eps, (maxx - minx));
	end
end


function exportFigureSafe(figHandle, outFile)
% exportgraphics() is newer; keep a fallback.
	[folder,~,~] = fileparts(outFile);
	if ~isfolder(folder)
		mkdir(folder);
	end

	[~,~,ext] = fileparts(outFile);
	ext = lower(ext);

	if exist('exportgraphics', 'file')
		try
			exportgraphics(figHandle, outFile, 'BackgroundColor', 'white', 'Resolution', 300);
			return;
		catch
			% Some MATLAB versions only accept axes/chart objects for exportgraphics.
			try
				ax = findall(figHandle, 'Type', 'axes');
				if ~isempty(ax)
					exportgraphics(ax(end), outFile, 'BackgroundColor', 'white', 'Resolution', 300);
					return;
				end
			catch
			end
		end
	end

	try
		set(figHandle, 'PaperPositionMode', 'auto');
	catch
	end
	switch ext
		case '.png'
			print(figHandle, outFile, '-dpng', '-r300');
		case {'.jpg', '.jpeg'}
			print(figHandle, outFile, '-djpeg', '-r300');
		case {'.tif', '.tiff'}
			print(figHandle, outFile, '-dtiff', '-r300');
		case '.pdf'
			print(figHandle, outFile, '-dpdf', '-painters');
		case '.eps'
			print(figHandle, outFile, '-depsc2', '-painters');
		otherwise
			saveas(figHandle, outFile);
	end
end


function exportFigureBundle(figHandle, outFile, cfg, domain)
	% Exports the requested raster/vector file AND (optionally) an editable .fig.
	% domain: 'paper' or 'output'
	if nargin < 4 || isempty(domain)
		domain = 'paper';
	end

	exportFigureSafe(figHandle, outFile);

	[folder, base, ~] = fileparts(outFile);
	if ~isfolder(folder)
		mkdir(folder);
	end

	% Save editable figure.
	saveEditable = false;
	exportPDF = false;
	if strcmpi(domain, 'paper')
		if isfield(cfg, 'paperFigs')
			if isfield(cfg.paperFigs, 'saveEditableFig'), saveEditable = cfg.paperFigs.saveEditableFig; end
			if isfield(cfg.paperFigs, 'exportPDF'), exportPDF = cfg.paperFigs.exportPDF; end
		end
	else
		if isfield(cfg, 'output') && isfield(cfg.output, 'saveEditableFig')
			saveEditable = cfg.output.saveEditableFig;
		end
	end

	if saveEditable
		try
			figFile = fullfile(folder, [base '.fig']);
			if exist('savefig', 'file')
				savefig(figHandle, figFile);
			else
				saveas(figHandle, figFile);
			end
		catch
		end
	end

	if exportPDF
		try
			pdfFile = fullfile(folder, [base '.pdf']);
			exportFigureSafe(figHandle, pdfFile);
		catch
		end
	end
end


function applyPRPlotScaling(recall, precision, cfg, isPerClass)
	% Plots a PR curve with optional scaling so near-1 curves are readable.
	if nargin < 4
		isPerClass = false;
	end

	scaleMode = 'full';
	topMargin = 0.02;
	minSpan = 0.10;
	if isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'perClassPR')
		if isfield(cfg.paperFigs.perClassPR, 'scaleMode'), scaleMode = char(cfg.paperFigs.perClassPR.scaleMode); end
		if isfield(cfg.paperFigs.perClassPR, 'topMargin'), topMargin = cfg.paperFigs.perClassPR.topMargin; end
		if isfield(cfg.paperFigs.perClassPR, 'minSpan'), minSpan = cfg.paperFigs.perClassPR.minSpan; end
	end

	precision = precision(:);
	recall = recall(:);
	mask = isfinite(precision) & isfinite(recall);
	precision = precision(mask);
	recall = recall(mask);

	if isempty(precision) || isempty(recall)
		plot(0, 0, '.'); grid on; ylim([0 1]);
		return;
	end

	if isPerClass && strcmpi(scaleMode, 'auto')
		if median(precision) > 0.85
			scaleMode = 'top';
		else
			scaleMode = 'full';
		end
	end

	switch lower(scaleMode)
		case 'log1mp'
			% Shows tiny differences when precision is close to 1.
			semilogy(recall, max(eps, 1 - precision), 'LineWidth', 1.5);
			grid on;
			ylim([1e-4 1]);
		case 'top'
			plot(recall, precision, 'LineWidth', 1.5);
			grid on;
			pMin = min(precision);
			yMin = max(0, pMin - topMargin);
			% Ensure some vertical span so the curve doesn't look flat.
			if (1 - yMin) < minSpan
				yMin = max(0, 1 - minSpan);
			end
			ylim([yMin 1]);
		case 'full'
			plot(recall, precision, 'LineWidth', 1.5);
			grid on;
			ylim([0 1]);
		otherwise
			% Fallback.
			plot(recall, precision, 'LineWidth', 1.5);
			grid on;
			ylim([0 1]);
	end
end


function scaleMode = getPerClassPRScaleMode(cfg)
	% Returns the configured per-class PR scaling mode.
	scaleMode = 'full';
	if isfield(cfg, 'paperFigs') && isfield(cfg.paperFigs, 'perClassPR') && isfield(cfg.paperFigs.perClassPR, 'scaleMode')
		scaleMode = char(cfg.paperFigs.perClassPR.scaleMode);
	end
end


function targetLongSide = getTargetLongSide(cfg)
	% Returns the desired maximum long-side length in pixels for processing.
	% This implements presets like "720p" and "480p".
	targetLongSide = Inf;
	if ~isfield(cfg, 'resize')
		return;
	end
	if isfield(cfg.resize, 'preset')
		switch lower(string(cfg.resize.preset))
			case "native"
				targetLongSide = Inf;
			case "2160p"
				targetLongSide = 3840;
			case "4k"
				targetLongSide = 3840;
			case "1440p"
				targetLongSide = 2560;
			case "2k"
				targetLongSide = 2560;
			case "1080p"
				targetLongSide = 1920;
			case "720p"
				targetLongSide = 1280;
			case "480p"
				targetLongSide = 854;
			case "360p"
				targetLongSide = 640;
			case "240p"
				targetLongSide = 426;
			otherwise
				% 'custom' or unknown preset
				if isfield(cfg.resize, 'targetLongSide')
					targetLongSide = cfg.resize.targetLongSide;
				end
		end
	elseif isfield(cfg.resize, 'targetLongSide')
		targetLongSide = cfg.resize.targetLongSide;
	end

	if isempty(targetLongSide) || ~isfinite(targetLongSide) || targetLongSide <= 0
		targetLongSide = Inf;
	end
end


function [scale, newSize] = computeScaleForPair(sizeA, sizeB, targetLongSide)
	% Computes a shared isotropic scale factor so that both images fit within
	% targetLongSide (based on the maximum of their long sides).
	Ha = sizeA(1); Wa = sizeA(2);
	Hb = sizeB(1); Wb = sizeB(2);
	longSide = max([Ha, Wa, Hb, Wb]);
	if isempty(targetLongSide) || ~isfinite(targetLongSide)
		scale = 1.0;
	else
		scale = min(1.0, targetLongSide / max(1, longSide));
	end
	newSize = [max(1, round(Ha * scale)), max(1, round(Wa * scale))];
end


function b2 = scaleBboxes(b, s)
	% Scales [x y w h] boxes by factor s (isotropic).
	if isempty(b)
		b2 = b;
		return;
	end
	b2 = b;
	b2(:, 1:4) = b(:, 1:4) * s;
end


function r2 = scaleRect(r, s)
	% Scales [x y w h] rectangle by factor s (isotropic).
	if isempty(r)
		r2 = r;
		return;
	end
	r2 = r * s;
end

function [roiMask, roiRect, padPx] = buildAvoidanceRoi(imgSize, cfg)
% Builds an inner-ROI mask that ignores a symmetric outer border.
% Padding is controlled by cfg.roi.avoidAreaRatio (fraction of image area to ignore).
% Returns roiMask, roiRect ([x y w h]), and padPx.

	H = imgSize(1);
	W = imgSize(2);

	if ~isfield(cfg, 'roi') || ~isfield(cfg.roi, 'enable') || ~cfg.roi.enable
		padPx = 0;
		roiMask = true(H, W);
		roiRect = [1, 1, W, H];
		return;
	end

	p = cfg.roi.avoidAreaRatio;
	p = max(0, min(0.95, p));

	% Convert area ratio -> symmetric padding thickness (solve for m).
	disc = (W + H)^2 - 4 * p * W * H;
	disc = max(0, disc);
	m = ((W + H) - sqrt(disc)) / 4; % smaller root
	padPx = floor(max(0, m));

	% Clamp if too aggressive
	padPx = min(padPx, floor(min(W, H) / 2) - 1);
	if padPx < 0
		padPx = 0;
	end

	x1 = 1 + padPx;
	y1 = 1 + padPx;
	x2 = W - padPx;
	y2 = H - padPx;

	roiMask = false(H, W);
	roiMask(y1:y2, x1:x2) = true;

	roiRect = [x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)];
end
