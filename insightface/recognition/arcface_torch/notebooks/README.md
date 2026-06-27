# Notebooks

Thư mục này lưu notebook và percent-format source dùng cho Colab/Kaggle.

```text
notebooks/
|-- colab_lightweight_fr.py
|-- colab_lightweight_fr.ipynb
|-- colab_phase2_loss_comparison.py
|-- colab_phase2_loss_comparison.ipynb
|-- colab_phase2_resume_runner.py
|-- colab_phase2_resume_runner.ipynb
|-- convert_to_ipynb.py
|-- convert_to_ipynb_phase2.py
`-- lightweight_kaggle_report.ipynb
```

Convert percent-format source sang notebook:

```bash
cd insightface/recognition/arcface_torch/notebooks
python convert_to_ipynb.py
python convert_to_ipynb_phase2.py
```

Notebook có thể chứa đường dẫn hoặc output từ môi trường Colab/Kaggle cũ; trước
khi chạy lại cần kiểm tra cell cấu hình dataset/checkpoint.

