import argparse
# import os.path
# import os
import os.path
from setfit import SetFitModel, SetFitTrainer
import pandas as pd
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.model_selection import train_test_split
from datasets import Dataset
import argparse


def main():
    parser = argparse.ArgumentParser(description="setfit binary clf")
    parser.add_argument('--mode', type=str, required=True, choices=['train','inference'])
    parser.add_argument('--train_data_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--num_epochs_body', type=int, default=1)
    parser.add_argument('--num_epochs_head', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--log_path', type=str, default='logs/log.txt', help='경로로 로그 파일을 저장합니다.')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode == 'train':
        result = train(args)
        return result
    # elif args.mode == 'inference':
    #     inference(args)

# def _tokenizer():

def dataLoad(args):
    global num_classes
    data_dir = args.train_data_dir
    df = pd.read_csv(data_dir + '/data.csv')
    if df.columns[0] != 'sentence' or df.columns[1] != 'label':
        raise ValueError("첫 번째 열은 'sentence'이어야 하고, 두 번째 열은 'label'이어야 합니다.")
    num_classes = len(df['label'].unique())
    train, eval = train_test_split(df, test_size=0.2, random_state=2023)
    train = train.reset_index(drop=True)
    eval = eval.reset_index(drop=True)
    train_ds = Dataset.from_pandas(train)
    eval_ds = Dataset.from_pandas(eval)
    return train_ds, eval_ds

def modelLoad():
#     model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
    return SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2",
                                       use_differentiable_head=True,
                                       head_params={"out_features": num_classes})

def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    model_id = params.get("model_id", "sentence-transformers/all-mpnet-base-v2")
    model_params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained(model_id, **model_params)

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [32]),
        "num_iterations": trial.suggest_categorical("num_iterations", [1, 2, 5, 10, 20]),
        "seed": trial.suggest_int("seed", 1, 40),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
        "model_id": trial.suggest_categorical(
            "model_id",
            [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L12-v1",
            ],
        ),
    }

def train(args):
    train_ds, eval_ds = dataLoad(args)
    model = modelLoad()
    trainer = SetFitTrainer(
        model=model,
        # model_init=model_init,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        # num_epochs=args.num_epochs,
        column_mapping={"sentence": "text", "label": "label"}
    )
    # freeze final layer
    print(f'#########  body part start ##################################')
    trainer.freeze()
    trainer.train(body_learning_rate=1e-5, num_epochs=args.num_epochs_body)
    print(f'#########  body part end ##################################')


    # unfreeze final layer
    print(f'#########  head part start ##################################')
    trainer.unfreeze(keep_body_frozen=True)
    trainer.train(learning_rate=1e-2, num_epochs=args.num_epochs_head)
    print(f'#########  head part end ##################################')

    metrics = trainer.evaluate()
    print(f'metrics : {metrics} ##################################')

    # best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=20)
    # print(f'best_run : {best_run} ################################')
    # trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    # trainer.train()
    # metrics = trainer.evaluate()
    # print(f'metrics : {metrics} ##################################')

# def inference():
if __name__ == '__main__':
    result = main()