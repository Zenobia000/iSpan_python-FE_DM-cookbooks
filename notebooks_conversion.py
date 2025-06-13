import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import shutil

# 指定要轉換的目錄
source_dir = r"D:\github\iSpan_python-FE_DM-cookbooks\data_mining_course\modules\module_01_eda_intro\notebooks"

# 遍歷目錄中的所有 .py 檔案
for filename in os.listdir(source_dir):
    if filename.endswith(".py"):
        py_file_path = os.path.join(source_dir, filename)
        
        # 讀取 py 檔案
        with open(py_file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # 根據 `# %%` 進行區隔
        cells_raw = code.split("# %%")
        cells = [new_code_cell(cell.strip()) for cell in cells_raw if cell.strip()]

        # 建立 notebook
        nb = new_notebook(cells=cells)

        # 儲存為 ipynb
        ipynb_file_path = py_file_path.replace(".py", ".ipynb")
        with open(ipynb_file_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

# def create_directory_structure():
#     """建立資料探勘課程的完整資料夾結構"""
    
#     # 基礎目錄
#     base_dir = "data_mining_course"
#     if os.path.exists(base_dir):
#         shutil.rmtree(base_dir)
#     os.makedirs(base_dir)
    
#     # 主要資料夾
#     main_dirs = [
#         "modules",
#         "datasets/raw",
#         "datasets/processed",
#         "utils",
#         "templates",
#         "projects/midterm",
#         "projects/final",
#         "environment",
#         "environment/docker",
#         "docs"
#     ]
    
#     for dir_path in main_dirs:
#         os.makedirs(os.path.join(base_dir, dir_path))
    
#     # 模組資料夾與子資料夾
#     modules = [
#         "module_01_eda_intro",
#         "module_02_data_cleaning",
#         "module_03_missing_outliers",
#         "module_04_categorical_encoding",
#         "module_05_scaling_transformation",
#         "module_06_feature_creation",
#         "module_07_feature_selection",
#         "module_08_time_series",
#         "module_09_multimodal_features",
#         "module_10_data_mining_applications"
#     ]
    
#     module_subdirs = ["slides", "notebooks", "exercises", "resources"]
    
#     for module in modules:
#         module_path = os.path.join(base_dir, "modules", module)
#         os.makedirs(module_path)
        
#         for subdir in module_subdirs:
#             os.makedirs(os.path.join(module_path, subdir))
    
#     # 特殊結構 - 模組9的子資料夾
#     multimodal_subdirs = [
#         "notebooks/01_text_features",
#         "notebooks/02_image_features",
#         "notebooks/03_audio_features"
#     ]
    
#     for subdir in multimodal_subdirs:
#         os.makedirs(os.path.join(base_dir, "modules", "module_09_multimodal_features", subdir))
    
#     # 特殊結構 - 模組10的子資料夾
#     datamining_subdirs = [
#         "notebooks/01_association_rules",
#         "notebooks/02_clustering",
#         "notebooks/03_tree_models"
#     ]
    
#     for subdir in datamining_subdirs:
#         os.makedirs(os.path.join(base_dir, "modules", "module_10_data_mining_applications", subdir))
    
#     # 資料集資料夾
#     datasets = [
#         "house_prices",
#         "titanic",
#         "insurance",
#         "nyc_taxi",
#         "breast_cancer",
#         "power_consumption",
#         "imdb_reviews",
#         "dogs_vs_cats",
#         "urban_sound",
#         "instacart",
#         "mall_customers",
#         "telco_churn"
#     ]
    
#     for dataset in datasets:
#         os.makedirs(os.path.join(base_dir, "datasets", "raw", dataset))
#         os.makedirs(os.path.join(base_dir, "datasets", "processed", dataset))
    
#     # 建立基本檔案
#     base_files = {
#         "README.md": "# 資料探勘與特徵工程課程\n\n本課程專注於資料探勘與特徵工程技術，從基礎EDA到進階特徵工程。",
#         "docs/syllabus.md": "# 課程大綱\n\n## 課程目標\n\n本課程旨在培養學生在資料探勘與特徵工程方面的實務能力。",
#         "docs/schedule.md": "# 課程時間表\n\n## 第一週\n- 模組1：課程導入與EDA複習",
#         "docs/references.md": "# 參考資料\n\n## 書籍\n- Feature Engineering for Machine Learning (O'Reilly)",
#         "docs/faq.md": "# 常見問題解答\n\n## 環境設定\n**Q: 如何安裝所需套件？**\nA: 請參考environment資料夾中的說明。",
#         "environment/requirements.txt": "pandas==2.0.0\nnumpy==1.24.3\nscikit-learn==1.3.0\nmatplotlib==3.7.1\nseaborn==0.12.2",
#         "environment/environment.yml": "name: data-mining\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.9\n  - pandas=2.0.0",
#         "utils/data_loader.py": "import pandas as pd\n\ndef load_dataset(dataset_name, processed=False):\n    \"\"\"載入指定的資料集\"\"\"\n    base_path = '../datasets/'\n    if processed:\n        base_path += 'processed/'\n    else:\n        base_path += 'raw/'\n    \n    # 根據資料集名稱返回對應的資料\n    if dataset_name == 'titanic':\n        return pd.read_csv(f'{base_path}titanic/train.csv')\n    elif dataset_name == 'house_prices':\n        return pd.read_csv(f'{base_path}house_prices/train.csv')\n    # 其他資料集...\n    else:\n        raise ValueError(f'未知的資料集: {dataset_name}')\n",
#         "utils/visualization.py": "import matplotlib.pyplot as plt\nimport seaborn as sns\nimport numpy as np\n\ndef plot_missing_values(df):\n    \"\"\"繪製缺失值熱圖\"\"\"\n    plt.figure(figsize=(10, 6))\n    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')\n    plt.title('缺失值分布圖')\n    plt.tight_layout()\n    return plt\n",
#         "utils/preprocessing.py": "import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\n\ndef handle_missing_values(df, strategy='mean'):\n    \"\"\"處理缺失值\"\"\"\n    if strategy == 'drop':\n        return df.dropna()\n    elif strategy == 'mean':\n        return df.fillna(df.mean())\n    elif strategy == 'median':\n        return df.fillna(df.median())\n    elif strategy == 'mode':\n        return df.fillna(df.mode().iloc[0])\n    else:\n        raise ValueError(f'不支援的策略: {strategy}')\n",
#         "utils/evaluation.py": "import numpy as np\nfrom sklearn.metrics import mean_squared_error, accuracy_score\n\ndef evaluate_regression(y_true, y_pred):\n    \"\"\"評估迴歸模型\"\"\"\n    mse = mean_squared_error(y_true, y_pred)\n    rmse = np.sqrt(mse)\n    return {\n        'MSE': mse,\n        'RMSE': rmse\n    }\n",
#         "templates/notebook_template.ipynb": "{\n \"cells\": [\n  {\n   \"cell_type\": \"markdown\",\n   \"metadata\": {},\n   \"source\": [\n    \"# 標題\\n\",\n    \"\\n\",\n    \"## 學習目標\\n\",\n    \"\\n\",\n    \"- 目標1\\n\",\n    \"- 目標2\\n\",\n    \"\\n\",\n    \"## 導入套件\"\n   ]\n  },\n  {\n   \"cell_type\": \"code\",\n   \"execution_count\": null,\n   \"metadata\": {},\n   \"outputs\": [],\n   \"source\": [\n    \"import pandas as pd\\n\",\n    \"import numpy as np\\n\",\n    \"import matplotlib.pyplot as plt\\n\",\n    \"import seaborn as sns\\n\",\n    \"\\n\",\n    \"# 設定視覺化風格\\n\",\n    \"plt.style.use('seaborn-v0_8')\\n\",\n    \"sns.set(font_scale=1.2)\\n\",\n    \"\\n\",\n    \"# 忽略警告訊息\\n\",\n    \"import warnings\\n\",\n    \"warnings.filterwarnings('ignore')\"\n   ]\n  }\n ],\n \"metadata\": {\n  \"kernelspec\": {\n   \"display_name\": \"Python 3\",\n   \"language\": \"python\",\n   \"name\": \"python3\"\n  },\n  \"language_info\": {\n   \"codemirror_mode\": {\n    \"name\": \"ipython\",\n    \"version\": 3\n   },\n   \"file_extension\": \".py\",\n   \"mimetype\": \"text/x-python\",\n   \"name\": \"python\",\n   \"nbconvert_exporter\": \"python\",\n   \"pygments_lexer\": \"ipython3\",\n   \"version\": \"3.9.7\"\n  }\n }\n}\n",
#         "templates/project_template.ipynb": "{\n \"cells\": [\n  {\n   \"cell_type\": \"markdown\",\n   \"metadata\": {},\n   \"source\": [\n    \"# 專案標題\\n\",\n    \"\\n\",\n    \"## 專案概述\\n\",\n    \"\\n\",\n    \"本專案旨在...\\n\",\n    \"\\n\",\n    \"## 資料集描述\"\n   ]\n  },\n  {\n   \"cell_type\": \"code\",\n   \"execution_count\": null,\n   \"metadata\": {},\n   \"outputs\": [],\n   \"source\": [\n    \"import pandas as pd\\n\",\n    \"import numpy as np\\n\",\n    \"import matplotlib.pyplot as plt\\n\",\n    \"import seaborn as sns\\n\",\n    \"\\n\",\n    \"# 設定視覺化風格\\n\",\n    \"plt.style.use('seaborn-v0_8')\\n\",\n    \"sns.set(font_scale=1.2)\\n\",\n    \"\\n\",\n    \"# 忽略警告訊息\\n\",\n    \"import warnings\\n\",\n    \"warnings.filterwarnings('ignore')\"\n   ]\n  }\n ],\n \"metadata\": {\n  \"kernelspec\": {\n   \"display_name\": \"Python 3\",\n   \"language\": \"python\",\n   \"name\": \"python3\"\n  },\n  \"language_info\": {\n   \"codemirror_mode\": {\n    \"name\": \"ipython\",\n    \"version\": 3\n   },\n   \"file_extension\": \".py\",\n   \"mimetype\": \"text/x-python\",\n   \"name\": \"python\",\n   \"nbconvert_exporter\": \"python\",\n   \"pygments_lexer\": \"ipython3\",\n   \"version\": \"3.9.7\"\n  }\n }\n}\n",
#         "projects/midterm/requirements.md": "# 期中專案需求\n\n## 目標\n\n選擇一個資料集，進行完整的特徵工程流程，包括：\n\n1. 資料清理與預處理\n2. 缺失值與異常值處理\n3. 特徵編碼與轉換\n4. 特徵創造與選擇",
#         "projects/midterm/evaluation_criteria.md": "# 評分標準\n\n## 技術實現 (40%)\n- 資料清理與預處理的完整性\n- 特徵工程技術的適當應用\n- 程式碼品質與效率",
#         "projects/final/requirements.md": "# 期末專案需求\n\n## 目標\n\n解決一個實際業務問題，完成從資料探勘到模型建立的完整流程：\n\n1. 問題定義與資料收集\n2. 資料探索與視覺化\n3. 特徵工程與選擇\n4. 模型建立與評估\n5. 結果解釋與業務建議",
#         "projects/final/evaluation_criteria.md": "# 評分標準\n\n## 問題解決 (30%)\n- 問題定義的清晰度\n- 解決方案的適當性\n- 業務價值的展現",
#         "environment/docker/Dockerfile": "FROM python:3.9-slim\n\nWORKDIR /app\n\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n\nCOPY . .\n\nCMD [\"jupyter\", \"lab\", \"--ip=0.0.0.0\", \"--port=8888\", \"--no-browser\", \"--allow-root\"]\n",
#         "environment/docker/docker-compose.yml": "version: '3'\nservices:\n  jupyter:\n    build: .\n    ports:\n      - \"8888:8888\"\n    volumes:\n      - ../..:/app\n    environment:\n      - JUPYTER_ENABLE_LAB=yes\n"
#     }
    
#     for file_path, content in base_files.items():
#         with open(os.path.join(base_dir, file_path), 'w', encoding='utf-8') as f:
#             f.write(content)
    
#     print("資料夾結構建立完成！")

# if __name__ == "__main__":
#     create_directory_structure()