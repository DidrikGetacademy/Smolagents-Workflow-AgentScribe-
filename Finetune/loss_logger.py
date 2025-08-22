from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments, Trainer
from datetime import datetime
from math import exp
class LossAndEvalloggingCallback(TrainerCallback):
    def __init__(self):
        self.prev_train_loss = None
        self.prev_eval_loss = None 
        self.last_logged_train_loss = None 


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

    


        token_acc_percent = f"{token_acc * 100:.2f}%" if token_acc is not None else "N/A"

        #Tabell header
        if step == 1 or step % 100 == 0:
            header = (
                f"\n{'Step':>5} | {'Loss':>7} | {'Perplexity':>10} | {'Grad Norm':>10} | {'Token Acc':>10} | {'Kommentar'}\n"
                + "-" * 75
            )
        else:
            header = ""

        row = f"{step:5} | {loss:.4f} | {perplexity:10.4f} | {grad_norm:10.4f} | {token_acc_percent:>9} "

        explanation = f"""
                        📘 Forklaringer:
                        🔹 Loss        – Mål på feil; lavere = bedre.
                        🔹 Perplexity  – exp(loss); hvor "forvirret" modellen er (nær 1 = bra).
                        🔹 Grad Norm   – Hvor kraftige oppdateringer modellen gjør (indikator på læring/stabilitet).
                        🔹 Token Acc   – Hvor mange tokens som ble korrekt forutsagt (%).
                        🔹 Kommentar   – Rask vurdering basert på loss-endring.
                        """
        
        log_msg = f"{header}\n{row}\n{explanation if step % 200 == 0 else ''}"

        log_msg = (
            f"[----------------TRAINING METRICS--------------]"
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] TRAIN | Step: {step} | Epoch: {epoch:.2f} | Samples: {samples_seen}\n"
            f"🔹 Loss: {f'{loss:.4f}' if loss is not None else 'N/A'} | Perplexity: {perplexity}\n"
            f"🔹 LR: {lr}\n"
            f"🔹 Grad Norm: {grad_norm}\n"
            f"🔹 #Tokens: {num_tokens}\n"
            f"🔹 Token Acc: {token_acc}\n\n"
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
            self.last_logged_train_loss = logs["loss"]

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        step = state.global_step
        epoch = state.epoch or 0.0
        eval_perplexity = exp(eval_loss) if eval_loss is not None else "N/A"

   
        if eval_loss is not None:
            delta = eval_loss - self.prev_eval_loss if self.prev_eval_loss is not None else "N/A"
            self.prev_eval_loss = eval_loss

            if delta is None:
                eval_comment = "🚀 Første eval-runde – bruker som referanse."
            elif delta < -0.05:
                eval_comment = "✅ Eval loss gikk tydelig ned – modellen lærer fortsatt godt."
            elif delta < -0.01:
                eval_comment = "📉 Eval loss litt lavere – forbedring."
            elif abs(delta) < 0.005:
                eval_comment = "😐 Eval loss står omtrent stille – kanskje metningspunkt?"
            elif delta > 0.05:
                eval_comment = "⚠️ Eval loss øker mye – mulig overfitting eller dårlig data."
            else:
                eval_comment = "🔄 Eval loss litt opp – følg med på trend og overtrening."
        else:
            eval_comment = "ℹ️ Ingen eval-loss logget."


        header = (
            f"\n{'Step':>5} | {'Eval Loss':>10} | {'Perplexity':>10} | {'Kommentar'}\n"
            + "-" * 60
        )
        row = f"{step:5} | "
        row += f"{eval_loss:.4f}" if eval_loss is not None else "   N/A   "
        row += "    | "
        row += f"{eval_perplexity:10.4f}" if isinstance(eval_perplexity, (float,int)) else "      N/A   "
        row += f" | {eval_comment}"


      
        explanation = f"""
                    📘 Eval Forklaringer:
                    🔹 Eval Loss   – Mål på feil på valideringsdata. Viktig for å oppdage overfitting.
                    🔹 Perplexity  – exp(eval_loss); nær 1 betyr lite usikkerhet.
                    🔹 Kommentar   – Tolker endring i eval-loss siden forrige runde.
                    """


        train_loss = self.prev_train_loss
        fit_comment = ""
        if train_loss is not None and eval_loss is not None:
            loss_gap = eval_loss - train_loss
            if loss_gap > 0.4:
                fit_comment = (
                    "\n🧠 Overfitting mistenkes: Eval loss mye høyere enn train loss."
                    "\n🔧 Tiltak: Prøv mer regularisering, mindre modell, mer data eller early stopping."
                )
            elif loss_gap < -0.2:
                fit_comment = (
                    "\n🤔 Eval loss lavere enn train loss – kanskje eval-sett er enklere?"
                    "\n🔍 Undersøk eval-data for skjevheter eller 'lekkasjer'."
                )
            else:
                fit_comment = "\n✅ Eval loss og train loss er balanserte – modellen ser ut til å generalisere godt."

    
        full_log = (
            f"[----------------EVALUATION METRICS--------------]"
            f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}]  EVAL  | Step: {step} | Epoch: {epoch:.2f}"
            f"\n{header}\n{row}\n{explanation}"
            f"{fit_comment}"
            f"\n{'-'*60}\n"
        )

        print(full_log)
        with open(
            r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\finetuning_loss_logg.txt",
            "a",
            encoding="utf-8"
        ) as f:
            f.write(full_log + "\n")
