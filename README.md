# DGCNN-PyTorch: Dynamic Graph CNN for Learning on Point Clouds

A **PyTorch implementation** of the **Dynamic Graph CNN (DGCNN)** for learning on point clouds. This repository covers tasks such as classification, part segmentation, and semantic segmentation.

![DGCNN Architecture](https://github.com/antao97/dgcnn.pytorch/blob/master/image/DGCNN.jpg)

---

## 🚀 Features

- **Point Cloud Classification**: Implemented on the ModelNet40 dataset.
- **Point Cloud Part Segmentation**: Implemented on the ShapeNet Part dataset.
- **Point Cloud Semantic Segmentation**: Implemented on the S3DIS and ScanNet datasets.
- Utilizes **dynamic graph construction** with k-nearest neighbors (**k-NN**).
- Efficient batch processing using **PyTorch**.

---

## 📂 Datasets

Due to size constraints, datasets are not included in this repository. Please download them from the provided links and place them in the appropriate directories as specified.

### 1️⃣ ModelNet40 Dataset

- **Description**: Contains 3D CAD models for object classification tasks.
- **Download Link**: [ModelNet40](https://modelnet.cs.princeton.edu/)
- **Preparation**:
  - After downloading, extract the files and place them in the `data/modelnet40/` directory.
  - Ensure the directory structure is as follows:
    ```
    data/
      modelnet40/
        modelnet40_train/
        modelnet40_test/
    ```

### 2️⃣ ShapeNet Part Dataset

- **Description**: Provides 3D models with part annotations for segmentation tasks.
- **Download Link**: [ShapeNet Part](https://www.shapenet.org/)
- **Preparation**:
  - Extract the downloaded files to the `data/shapenet_part/` directory.
  - Directory structure:
    ```
    data/
      shapenet_part/
        train_data/
        val_data/
        test_data/
    ```

### 3️⃣ S3DIS Dataset

- **Description**: Contains 3D scans of indoor scenes for semantic segmentation.
- **Download Link**: [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html)
- **Preparation**:
  - Place the extracted files in the `data/s3dis/` directory.
  - Directory structure:
    ```
    data/
      s3dis/
        Area_1/
        Area_2/
        ...
    ```

### 4️⃣ ScanNet Dataset

- **Description**: A large-scale dataset of indoor scenes for 3D object recognition.
- **Download Link**: [ScanNet Dataset](http://www.scan-net.org/)
- **Preparation**:
  - Extract and place the files in the `data/scannet/` directory.
  - Directory structure:
    ```
    data/
      scannet/
        scans/
        scans_test/
    ```

---

## 📦 Installation

Ensure you have Python **≥3.7** and PyTorch **≥1.2** installed.

```bash
pip install -r requirements.txt

## 📦 Dependencies

Ensure you have Python **≥3.7** and PyTorch **≥1.2** installed.

Install the required packages:

pip install -r requirements.txt

markdown
Copy
Edit

**Required Packages:**
- `torch`
- `torchvision`
- `numpy`
- `h5py`
- `scipy`
- `sklearn`
- `plyfile`
- `torch_scatter`

---

## 🏗 Usage

### 1️⃣ Point Cloud Classification

**Train the model on ModelNet40:**
python main_cls.py --exp_name=cls_1024 --num_points=1024 --k=20

css
Copy
Edit

**Evaluate a trained model:**
python main_cls.py --exp_name=cls_1024_eval --num_points=1024 --k=20 --eval=True --model_path=outputs/cls_1024/models/model.t7

shell
Copy
Edit

### 2️⃣ Point Cloud Part Segmentation
python main_partseg.py --exp_name=partseg --class_choice=airplane

shell
Copy
Edit

### 3️⃣ Point Cloud Semantic Segmentation (S3DIS)
python main_semseg_s3dis.py --exp_name=semseg_s3dis --test_area=6

yaml
Copy
Edit

---

## 📊 Results

### ModelNet40 Classification
| Method   | Accuracy |
|----------|----------|
| Paper    | 92.9%    |
| This Repo| 93.3%    |

### ShapeNet Part Segmentation
| Method   | mIoU     |
|----------|----------|
| Paper    | 85.2%    |
| This Repo| 85.2%    |

### S3DIS Semantic Segmentation
| Method   | mIoU     |
|----------|----------|
| Paper    | 56.1%    |
| This Repo| 59.2%    |

---

## 📜 Citation

If you use this code, please cite:
@article{wang2018dynamic, title={Dynamic Graph CNN for Learning on Point Clouds}, author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Saraswat, Sanjay and Bronstein, Michael and Solomon, Justin}, journal={arXiv preprint arXiv:1801.07829}, year={2018} }

yaml
Copy
Edit

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
