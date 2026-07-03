# FORC 项目 Baseline 搭建说明

## 项目目标

本 baseline 对应你的 project plan：用模拟 FORC diagrams 训练机器学习反演模型，再评估它迁移到真实实验 FORC 数据时的表现差距。

核心问题不是一开始就做复杂模型，而是先建立一个可靠的基准：

1. 模拟数据上能不能预测 grain-size distribution 或 domain-state 类别？
2. 模拟 FORC 和实验 FORC 的分布差异在哪里？
3. 噪声增强、强度增强、简单 fine-tuning 能不能缩小 simulation-to-real gap？
4. 哪种方法在准确率、可解释性和 IRP 时间限制内最合适？

## Baseline 包含什么

- `.csv` 和 `.frc` 文件读取入口；
- FORC 图统一插值到固定 2D grid；
- z-score/min-max 归一化；
- 小型 CNN 作为第一版模型；
- 噪声增强和强度缩放增强；
- train/validation/test 分割；
- MAE 和 RMSE 评估模板；
- 后续真实实验数据 qualitative/semi-quantitative 分析模板。

## 第一阶段应该怎么做

### 1. 整理数据

把模拟数据放到：

```text
data/raw/simulated/
```

把实验数据放到：

```text
data/raw/experimental/
```

再创建一个标签表：

```text
data/raw/simulated_labels.csv
```

最小格式示例：

```csv
filename,grain_size
sample_001.csv,0.35
sample_002.csv,0.80
```

如果你的任务是分类，可以把 `grain_size` 换成 `domain_state`，并在代码里把 loss 和 metric 改成分类版本。

### 1.5. 运行数据检查

先运行：

```powershell
.\run_stage1_inspection.ps1
```

它会生成：

```text
data/raw/simulated_labels_template.csv
reports/stage1_data_inspection.md
```

然后把 `simulated_labels_template.csv` 里的 `grain_size` 补上，并另存为：

```text
data/raw/simulated_labels.csv
```

### 2. 先跑模拟数据 baseline

编辑：

```text
configs/baseline.yaml
```

确认：

- `target_columns` 是你的标签列；
- `grid_size` 适合 FORC 图分辨率；
- `normalisation` 暂时用 `zscore`；
- `model.output_dim` 等于目标数量。

运行：

```powershell
.\run_baseline.ps1
```

### 3. 记录第一版结果

结果会写到：

```text
reports/runs/baseline/
```

你还可以把实验观察填到：

```text
reports/baseline_results.md
```

### 2.5. 预处理冒烟测试

在真实训练之前，可以先运行：

```powershell
.\run_stage2_smoke_test.ps1
```

它会生成一个很小的 synthetic FORC-like 数据集，并检查：

- `.csv` 是否能被读取；
- FORC 点数据是否能插值到统一 grid；
- 归一化是否正常；
- 是否能输出预处理图像。

输出位置：

```text
reports/stage2_preprocessing_smoke_test.md
reports/figures/stage2_smoke/
```

这一步只是验证 pipeline，不作为科研结果。

### 4. 再做 sim-to-real 改进

按优先级建议：

1. clean simulation baseline；
2. noise augmentation；
3. intensity/scale augmentation；
4. simulated vs experimental statistics comparison；
5. small real-data fine-tuning；
6. feature alignment。

## 和 project plan 的时间线对应

- Weeks 1-2：读文献，检查 FORC 数据文件夹；
- Week 3：确定 label/target，补齐 `simulated_labels.csv`；
- Week 4：完善 preprocessing pipeline；
- Weeks 5-6：训练并评估 baseline CNN；
- Week 7：分析模拟和实验 FORC 差异；
- Weeks 8-9：加入噪声增强、强度增强、可选 fine-tuning；
- Week 10：比较 baseline 和 adapted models；
- Week 11：整理图表、失败案例；
- Week 12：写 final report。

## 注意事项

- 真实实验数据可能没有完整 ground truth，所以 baseline 的第一目标是把模拟数据评估做扎实。
- 如果 `.frc` 格式和当前读取函数不匹配，优先修改 `src/forc_baseline/preprocess.py` 的 `read_forc_table()`。
- 不要太早做复杂 domain adaptation；先用简单增强方法建立可解释结果。
