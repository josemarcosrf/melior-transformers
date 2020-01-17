from multiprocessing import cpu_count


global_args = {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",
    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 128,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 8,
    "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 4e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "do_lower_case": False,
    "logging_steps": 50,
    "save_steps": 2000,
    "save_model_every_epoch": True,
    "evaluate_during_training": False,  # Only evaluate when single GPU otherwise metrics may not average well
    "evaluate_during_training_steps": 2000,
    "use_cached_eval_features": False,
    "tensorboard_dir": None,
    "overwrite_output_dir": False,
    "reprocess_input_data": False,
    "process_count": cpu_count() - 2 if cpu_count() > 2 else 1,
    "n_gpu": 1,
    "use_multiprocessing": True,
    "silent": False,
    # Melior paramas
    "metric_criteria": "f1",  # Could be f1, acc, precision or mcc
    "save_n_best_epochs": 1,  # Save just the N best models
    "sliding_window": False,
    "tie_value": 1,
    "stride": 0.8,
    "regression": False,
    "wandb_project": None,
    "wandb_kwargs": {},
}
