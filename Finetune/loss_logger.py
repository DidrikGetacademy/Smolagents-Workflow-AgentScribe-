from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments, Trainer
from datetime import datetime
from math import exp
class LossAndEvalloggingCallback(TrainerCallback):
    def __init__(self):
        self.prev_train_loss = None

    def _write_train_log(self, args, state, logs):
        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm", "N/A")
        lr = logs.get("learning_rate", "N/A")
        num_tokens = logs.get("num_tokens", "N/A")
        token_acc = logs.get("mean_token_accuracy", "N/A")
        step = state.global_step
        epoch = state.epoch or 0.0
        samples_seen = args.per_device_train_batch_size * args.gradient_accumulation_steps * step
        perplexity = exp(loss) if loss is not None else "N/A"

    
        if loss is not None:
            change = loss - self.prev_train_loss if self.prev_train_loss is not None else None
            if change is None:
                comment = "🚀 Starter opp!\nFørste loss – bruker som referanse fremover."
                color = "\033[94m"
            elif change < -0.1:
                comment = f"📉 Klar forbedring!\nLoss sank med {change:.4f} siden sist. God fremgang!"
                color = "\033[92m"
            elif change < -0.01:
                comment = f"✅ Litt bedre!\nLoss ned med {change:.4f}. Stabil læring."
                color = "\033[92m"
            elif abs(change) < 0.001:
                comment = f"😐 Står omtrent stille... ({change:.4f})\nMulig metningspunkt eller behov for justering."
                color = "\033[93m"
            elif change > 0.1:
                comment = f"⚠️ Loss øker mye! ({change:.4f})\nMulig overtrening, datastøy eller for høy læringsrate."
                color = "\033[91m"
            else:
                comment = f"🔄 Litt ustabilt... ({change:.4f})\nIkke alvorlig, men følg med på trenden."
                color = "\033[93m"
            self.prev_train_loss = loss
        else:
            comment = "ℹ️ Ingen loss logget dette steget."
            color = "\033[90m"

        log_msg = (
            f"[----------------TRAINING METRICS--------------]"
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] TRAIN | Step: {step} | Epoch: {epoch:.2f} | Samples: {samples_seen}\n"
            f"🔹 Loss: {f'{loss:.4f}' if loss is not None else 'N/A'} | Perplexity: {perplexity}\n"
            f"🔹 LR: {lr}\n"
            f"🔹 Grad Norm: {grad_norm}\n"
            f"🔹 #Tokens: {num_tokens}\n"
            f"🔹 Token Acc: {token_acc}\n\n"
            f"{comment}\n"
            + "-"*60
            + "\n\n\n"
        )
        print(log_msg)
        with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\finetuning_loss_logg.txt", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        # Kjør trenings‑logging når loss finnes
        if logs and logs.get("loss") is not None:
            self._write_train_log(args, state, logs)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        step = state.global_step
        epoch = state.epoch or 0.0
        eval_perplexity = exp(eval_loss) if eval_loss is not None else "N/A"

        comment = "ℹ️ Evaluerings‑runde fullført."
        log_msg = (
            f"[----------------EVALUATION METRICS--------------]"
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}]  EVAL  | Step: {step} | Epoch: {epoch:.2f}\n"
            f"🔹 Eval Loss: {f'{eval_loss:.4f}' if eval_loss is not None else 'N/A'} | Perplexity: {eval_perplexity}\n"
            f"{comment}\n"
            + "-"*60
            + "\n\n\n"
        )
        print(log_msg)
        with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\finetuning_loss_logg.txt", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
