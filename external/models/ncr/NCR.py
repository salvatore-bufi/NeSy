from tqdm import tqdm
import numpy as np
import torch
import os

from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .NCRModel import NCRModel


class NCR(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        '''
        _lr := learning rate
        _lw := regularization weight
        _rw := logical regularization weight
        _emb := dimension of user, item, and event embeddings
        _dropout := proportion of units that are randomly deactivated in each hidden layer while the model is undergoing training
        _max_history := maximum length of history for each interaction, i.e., max number of items at the left of the implication

        '''

        self._sampler = cs.Sampler(self._data.i_train_dict)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_lr", "lr", "lr", 0.0005, float, None),
            ("_rw", "rw", "rw", 0.1, float, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_emb", "emb", "emb", 64, int, None),
            ("_dropout", "dropout", "dropout"),
            ("_max_history", "max_history", "max_history", 5, int, None),
            ("_seed", "seed", "seed", 10, int, None),
            ("_loader", "loader", "loader", "KGTORETSVLoader", None, None)
        ]

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
        print("Using device:", device)

        self.autoset_params()
        self._side = getattr(self._data.side_information, self._loader, None)



        self._model = NCRModel(
            num_users=self._num_users,
            num_items=self._num_items,
            num_interactions=self.num_interactions,
            learning_rate=self._lr,
            edges_lr=self._elr,
            embedding_size=self._factors,
            l_w=self._l_w,
            alpha=self._alpha,
            n_layers=self._n_layers,
            edge_index=self.edge_index,
            item_features=self.item_features,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "KGUF" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(gu[offset: offset_stop], gi)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")
