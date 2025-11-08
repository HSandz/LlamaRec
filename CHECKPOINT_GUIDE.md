# Hướng dẫn Resume Training từ Checkpoint - Llama 7B

## 1. Checkpoint được lưu tự động

Khi fine-tune Llama 7B, Hugging Face Trainer tự động lưu checkpoint:

### Checkpoint Tự Động
- **Thư mục**: `experiments/llm/{dataset_code}/checkpoints/checkpoint-{step}/`
- **Lưu**: Mỗi `save_steps` (mặc định: mỗi 500 steps)
- **Nội dung**: 
  - Model weights (adapter_model.bin cho LoRA)
  - Optimizer state
  - Scheduler state
  - Training arguments
  - Trainer state (step, epoch, global_step)

### Best Model Checkpoint
- **Thư mục**: `experiments/llm/{dataset_code}/checkpoints/` (checkpoint tốt nhất)
- **Lưu**: Khi metric cải thiện (nếu có `load_best_model_at_end=True`)
- **Nội dung**: Model với metric validation tốt nhất

## 2. Cách Resume Training Llama 7B

### Resume tự động từ checkpoint gần nhất:

```bash
python train_ranker.py \
    --dataset_code beauty \
    --resume_from_checkpoint experiments/llm/beauty/checkpoints/checkpoint-1000
```

### Resume với dataset khác:

```bash
python train_ranker.py \
    --dataset_code games \
    --resume_from_checkpoint experiments/llm/games/checkpoints/checkpoint-500
```

### Resume với Yelp 2020:

```bash
python train_ranker.py \
    --dataset_code yelp2020 \
    --resume_from_checkpoint experiments/llm/yelp2020/checkpoints/checkpoint-2000
```

## 3. Cấu trúc Checkpoint Llama

Mỗi checkpoint folder chứa:

```
checkpoint-1000/
├── adapter_model.bin          # LoRA adapter weights
├── adapter_config.json        # LoRA configuration
├── optimizer.pt               # Optimizer state
├── scheduler.pt               # LR scheduler state
├── trainer_state.json         # Training state (epoch, step, loss)
├── training_args.bin          # Training arguments
└── rng_state.pth             # Random state for reproducibility
```

## 4. Lợi ích của Resume

- ✅ **Tiếp tục fine-tuning** nếu bị gián đoạn (OOM, crash, power loss)
- ✅ **Tiết kiệm thời gian** - không cần train lại từ đầu
- ✅ **Giữ nguyên optimizer state** - learning rate schedule, Adam momentum được bảo toàn
- ✅ **Reproducibility** - random state được lưu để training deterministic
- ✅ **Thử nghiệm hyperparameters** - load checkpoint và điều chỉnh learning rate, epochs

## 5. Tips cho Fine-tuning Local Llama 7B

### Liệt kê tất cả checkpoint:

```powershell
# Windows PowerShell
Get-ChildItem -Path experiments/llm/beauty/checkpoints/ -Directory

# Hoặc
dir experiments/llm/beauty/checkpoints/
```

### Tìm checkpoint gần nhất:

```powershell
# Sắp xếp theo thời gian tạo
Get-ChildItem experiments/llm/beauty/checkpoints/ | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### Resume với hyperparameters khác:

```bash
python train_ranker.py \
    --dataset_code beauty \
    --resume_from_checkpoint experiments/llm/beauty/checkpoints/checkpoint-1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2
```

### Kiểm tra training progress:

```powershell
# Đọc trainer_state.json để xem epoch, step hiện tại
Get-Content experiments/llm/beauty/checkpoints/checkpoint-1000/trainer_state.json | ConvertFrom-Json
```

## 6. Troubleshooting

### Lỗi: "Checkpoint not found"
- Kiểm tra đường dẫn có đúng không
- Đảm bảo checkpoint folder chứa file `adapter_model.bin` và `trainer_state.json`

### Lỗi: "CUDA out of memory"
- Giảm `per_device_train_batch_size`
- Tăng `gradient_accumulation_steps`
- Sử dụng `gradient_checkpointing=True`

### Resume nhưng train lại từ step 0
- Đảm bảo sử dụng đúng checkpoint path
- Kiểm tra file `trainer_state.json` có tồn tại không

### Model không cải thiện sau resume
- Checkpoint có thể bị corrupt, thử checkpoint cũ hơn
- Kiểm tra learning rate schedule có phù hợp không

## 7. Best Practices

### Cấu hình checkpointing tối ưu:

```python
# Trong trainer/llm.py hoặc config
training_args = TrainingArguments(
    output_dir="experiments/llm/beauty/checkpoints",
    save_strategy="steps",
    save_steps=500,                    # Lưu mỗi 500 steps
    save_total_limit=3,                # Chỉ giữ 3 checkpoint gần nhất
    load_best_model_at_end=True,       # Load best model sau training
    resume_from_checkpoint=args.resume_from_checkpoint,  # Support resume
)
```

### Backup checkpoint quan trọng:

```powershell
# Copy checkpoint tốt nhất ra ngoài
Copy-Item -Path "experiments/llm/beauty/checkpoints/checkpoint-1000" -Destination "backups/checkpoint-1000" -Recurse
```

### Monitoring disk space:

```powershell
# Kiểm tra dung lượng checkpoints
Get-ChildItem experiments/llm/ -Recurse | Measure-Object -Property Length -Sum
```

## 8. Ví dụ Workflow Hoàn Chỉnh

```bash
# 1. Bắt đầu training
python train_ranker.py --dataset_code beauty

# 2. Training bị gián đoạn ở step 1000

# 3. Resume từ checkpoint gần nhất
python train_ranker.py \
    --dataset_code beauty \
    --resume_from_checkpoint experiments/llm/beauty/checkpoints/checkpoint-1000

# 4. Training hoàn thành, best model được lưu tự động
```
