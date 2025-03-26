# 🚀 Optimized Clustering Algorithms for Text (GPU Accelerated)

A **Python-powered tool** for performing multiple clustering algorithms, dynamically merging results, and optimizing performance with GPU acceleration. The pipeline also includes advanced text preprocessing for enhanced accuracy.

---

## 📚 **Project Motivation and Overview**

### **Motivation**
Hello,

I undertook this project to demonstrate my ability to write standardized and robust Python code. I wanted to prove that I can develop a complete pipeline involving preprocessing (including feature engineering), algorithm optimization (utilizing different metrics), and generating results that are both insightful for business stakeholders and valuable for data scientists aiming to improve the models.

### **Process Overview**

#### **Preprocessing**
- The process begins with cleaning unstructured text data, which may include emojis, irregular indentations, and missing punctuation. These are transformed into a standardized format to ensure more robust BERT embeddings, enhancing clustering accuracy.
- Common sentences containing specific keywords are identified and removed to prevent skewed vectorization. For instance, repetitive sentences like "You can subscribe to the app for these prices" are erased by choosing essential keywords such as "subscribe." This significantly improves vectorization and clustering accuracy.

#### **Clustering**
- Multiple clustering algorithms are incorporated, allowing for flexible selection and application.
- Optimization techniques using metrics like the **Silhouette Score** are implemented, involving:
  - Iteratively testing different clustering strategies (varying cluster numbers and distance metrics).
  - Calculating the Silhouette Score for each iteration.
  - Returning the configuration with the best clustering performance.

#### **Merging Clustering Results**
- After generating multiple clustering results, they are dynamically merged to create a comprehensive final clustering. 
- This approach combines the strengths of different algorithms, leading to more optimal groupings. 
- Improvements are observable in the `merged_results.csv` file.

#### **Flexibility and Customization**
- All optimization parameters and configurations are adjustable via the `config.yaml` file, allowing for tailored customization to suit specific project requirements.

Thank you for reading! I hope this algorithm proves valuable for your projects. 🚀

---

## 📂 **Project Structure**

```
optimized_clustering_algos/
├── main/                     # Main execution scripts
│   └── main.py
├── multi_clustering/         # Clustering algorithm modules
│   ├── __init__.py
│   ├── kmeans.py
│   ├── gmm.py
│   ├── ... (other algorithms)
├── config.yaml                # Configuration for models and parameters
├── requirements.txt           # Python package requirements
├── environment.yml            # Conda environment for GPU dependencies
├── input_sheet1.csv           # Sample input CSV for trying out the project
├── .gitignore                 # Files to ignore in Git
└── README.md                  # This file
```

---

## 🔥 **Key Features**
- ✅ Multiple clustering algorithms: KMeans, GMM, Agglomerative, Spectral, etc.
- ✅ GPU-accelerated with **RAPIDS, CuPy, TensorFlow GPU, and PyTorch**.
- ✅ Dynamic merging of clustering results.
- ✅ Advanced text preprocessing with language detection, stopword removal, and SBERT embeddings.
- ✅ Customizable via `config.yaml` for models, parameters, and custom keywords.
- ✅ Clean, minimal final DataFrame with only essential information.
- ✅ Includes a sample `input_sheet1.csv` file to easily try out the project.

---

## 🚀 **Setup Instructions**

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/optimized_clustering_algos.git
cd optimized_clustering_algos
```

---

### 2. Create and Activate Conda Environment (for GPU)

```bash
conda env create -f environment.yml
conda activate optimized-clustering
```

> ✅ **Note:** Ensure you have compatible GPU drivers and CUDA installed.  
> Check GPU status with:

```bash
nvidia-smi
nvcc --version
```

---

### 3. Run the Main Script

```bash
python main/main.py --id_col name --text_column descriptions --min_common 3 --algorithms kmeans gmm agglomerative spectral --config config.yaml --input_file input_sheet1.csv
```

---

## ⚙️ **How the Pipeline Works**

1. **Preprocessing**  
   - Text is cleaned, normalized, and filtered based on custom keywords.  
   - Non-English texts are automatically filtered.  
   - SBERT embeddings are dynamically generated.  
   - Missing NLP models (like SpaCy and NLTK resources) are automatically downloaded.

2. **Clustering**  
   - Multiple algorithms (KMeans, GMM, Agglomerative, Spectral, etc.) process the embeddings.  
   - GPU acceleration ensures efficient processing.

3. **Merging**  
   - Clustering results are dynamically merged based on the `min_common` parameter.

4. **Final Output**  
   - A minimal CSV is generated with:
     - Unique Identifier (`name` or `app_id`)
     - `cleaned_text`
     - Individual clustering results
     - `merged_cluster`
     - `cluster_summary`

---

## ⚡ **GPU and CUDA Requirements**

- Ensure CUDA (>=11.8) is correctly installed.  
- Validate with:

```bash
nvidia-smi
nvcc --version
```

---

## 🐛 **Troubleshooting**

- **`CUDA Driver Not Found`**: Ensure the right version of CUDA is installed.  
- **`NVIDIA-SMI Not Found`**: Install NVIDIA GPU drivers.  
- **`No GPU Devices Available`**: Check GPU status with `nvidia-smi`.

---

## 🧹 **Cleaning Up**

To completely remove the environment:

```bash
conda deactivate
conda remove --name optimized-clustering --all
```

---

## 🤝 **Contributing**

Feel free to fork this repo, submit PRs, or raise issues.

---

## 📜 **License**

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 **Credits**

- [RAPIDS.ai](https://rapids.ai)  
- [SpaCy](https://spacy.io)  
- [Sentence-Transformers](https://www.sbert.net/)  
- [Scikit-learn](https://scikit-learn.org/)


## 🙌 **Credits**

- [RAPIDS.ai](https://rapids.ai)  
- [SpaCy](https://spacy.io)  
- [Sentence-Transformers](https://www.sbert.net/)  
- [Scikit-learn](https://scikit-learn.org/)
