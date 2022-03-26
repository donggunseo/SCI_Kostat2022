from transformers import Trainer
import datasets

class CustomTrainer(Trainer):
    def __init__(self, *args, eval_gt = None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_function = post_process_function
        self.eval_gt = eval_gt
        
    def evaluate(self, eval_dataset=None, eval_gt=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_gt = self.eval_gt  if eval_gt is None else eval_gt
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics
            
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )
        if self.post_process_function is not None and self.compute_metrics is not None:
            print("post-processing")
            eval_preds = self.post_process_function(output.predictions)
            print("metric computation")
            metrics = self.compute_metrics(eval_gt ,eval_preds)
            self.log(metrics)
        else:
            metrics = {}
            
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        
        return metrics