# Choquet Interval Ensemble for Motor Imager

The proposed prediction pipeline is the following:

![Prediction pipeline](/prediction_pipeline.png)

The intervalar Choquet integral is applied as follows:

![Intervalar Choquet integral](/int_choquet_integ.png)

## Table of contents

- [Choquet Interval Ensemble for Motor Imager](#choquet-interval-ensemble-for-motor-imager)
  - [Table of contents](#table-of-contents)
- [Project configuration](#project-configuration)
  - [Configuration variables](#configuration-variables)
  - [Environment variables](#environment-variables)
- [Commits guide](#commits-guide)

# Project configuration

## Configuration variables

The file [config.py](src/config.py) holds the value for various configuration variables used for the execution of the code. They can be modified as wished.

## Environment variables

The following variables should be defined in the environment where the code is running:

```
SAVE_TO_DISK: whether to save the results to disk or not (True|False)
SAVE_HDF5: whether to save the models to disk using HDF5 or not (True|False)
DISK_PATH: path where results are stored in disk (string)
```

If they are included in a [.env](.env) file in the root directory of the project, they will be automatically loaded.

# Commits guide

```
♻️ [Refactor code] ♻️
⚡️ [Improve code] ⚡️
🐛 [Fix bug] 🐛
🩹 [Fix non critical bug] 🩹
🚑️ [Critical Hotfix] 🚑️
✅ [Add feature] ✅
✨ [Add small feature] ✨
🩺 [Add sanity check] 🩺
🧪 [Add non tested feature]
🎉 [Feature passed a test]
🚧 [Feature not completed] 🚧
💥 [Add breaking changes] 💥
➕ [Add dependency] ➕
➖ [Remove dependency] ➖
🗑️ [Remove] 🗑️
📝 [Document] 📝
🔀 [Merge] 🔀
🎨 [Code formatting] 🎨
🔒️ [Fix security issues] 🔒️
```
