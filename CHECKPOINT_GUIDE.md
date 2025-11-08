# Hướng dẫn Resume Training từ Checkpoint

## 1. Checkpoint được lưu tự động

Trong quá trình training, hệ thống tự động lưu 2 loại checkpoint:

### a) Recent Checkpoint (Checkpoint gần nhất)
- **File**: `experiments/{model_code}/{dataset_code}/models/checkpoint-recent.pth`
- **Lưu**: Mỗi epoch
- **Nội dung**: Model state, optimizer state, epoch hiện tại

### b) Best Model Checkpoint (Model tốt nhất)
- **File**: `experiments/{model_code}/{dataset_code}/models/best_acc_model.pth`
- **Lưu**: Khi metric cải thiện
- **Nội dung**: Model state với metric tốt nhất

### c) Final Checkpoint
- **File**: `experiments/{model_code}/{dataset_code}/models/checkpoint-recent.pth.final`
- **Lưu**: Khi training hoàn thành

## 2. Cách Resume Training

### Resume từ checkpoint gần nhất:

```bash
python train_retriever.py \
    --dataset_code beauty \
    --resume_from_checkpoint experiments/lru/beauty/models/checkpoint-recent.pth
```

### Resume từ checkpoint cụ thể:

```bash
python train_retriever.py \
    --dataset_code ml-100k \
    --resume_from_checkpoint path/to/your/checkpoint.pth
```

### Resume cho LLM Ranker:

```bash
python train_ranker.py \
    --dataset_code games \
    --resume_from_checkpoint experiments/llm/games/models/checkpoint-recent.pth
```

## 3. Checkpoint Structure

Mỗi checkpoint chứa:

```python
{
    'model_state_dict': {...},      # Model weights
    'optimizer_state_dict': {...},  # Optimizer state (learning rate, momentum, etc)
    'epoch': 5,                      # Epoch number when saved
}
```

## 4. Lợi ích của Resume

- ✅ **Tiếp tục training** nếu bị gián đoạn (crashed, out of memory, etc)
- ✅ **Tiết kiệm thời gian** - không cần train lại từ đầu
- ✅ **Giữ nguyên optimizer state** - learning rate, momentum được bảo toàn
- ✅ **Thử nghiệm hyperparameters** - load checkpoint và thay đổi learning rate, batch size, etc

## 5. Tips

### Kiểm tra checkpoint có tồn tại:

```bash
# Windows PowerShell
Test-Path experiments/lru/beauty/models/checkpoint-recent.pth

# Linux/Mac
ls experiments/lru/beauty/models/
```

### Tìm checkpoint tốt nhất:

Checkpoint với metric tốt nhất luôn được lưu ở:
```
experiments/{model_code}/{dataset_code}/models/best_acc_model.pth
```

### Resume với hyperparameters khác:

```bash
python train_retriever.py \
    --dataset_code beauty \
    --resume_from_checkpoint experiments/lru/beauty/models/checkpoint-recent.pth \
    --lr 0.0005 \
    --num_epochs 50
```

## 6. Troubleshooting

### Lỗi: "Checkpoint not found"
- Kiểm tra đường dẫn có đúng không
- Đảm bảo đã train ít nhất 1 epoch để tạo checkpoint

### Lỗi: "Size mismatch"
- Checkpoint không tương thích với model hiện tại
- Đảm bảo sử dụng cùng model architecture

### Resume nhưng train lại từ epoch 0
- Kiểm tra checkpoint có chứa field 'epoch' không
- Đảm bảo sử dụng đúng checkpoint file
