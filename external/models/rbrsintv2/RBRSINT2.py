"""
Module description:

"""

import torch
import os
from tqdm import tqdm
import json

from elliot.dataset.samplers import custom_sampler as cs
from elliot.utils.write import store_recommendation

from elliot.recommender import BaseRecommenderModel
from .RBRSINT2Model import RBRSINT2Model
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger


class RBRSINT2(RecMixin, BaseRecommenderModel):
    r"""
    Rule Based RS - The rules in this version are SHARED

    C'è una matrice di "regole", da cui gli utenti tramite attention prelevano la regola.
    Gr = (num_regol, dim_emb)

    Avremo: per ogni utente nr embedding, ovvero ogni regola crea un embedding diverso !
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w]: regularization,
                                      lr: learning rate}
        """

        self._params_list = [
            ("_factors", "factors", "factors", 10, int, None),
            ("_lr", "lr", "lr", 0.001, float, None),
            ("_l_w", "l_w", "l_w", 0.1, float, None),
            ("_n_rules", "n_rules", "n_rules", 2, int, None),
            ("_eps", "eps", "eps", 1e-30, float, None),
            ("_l_rc", "l_rc", "lrc", 0.005, float, None),
            ("_nint", "nint", "nint", 32, int, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = cs.Sampler(self._data.i_train_dict, self._seed)

        self._model = RBRSINT2Model(num_users=self._num_users, num_items=self._num_items, lr=self._lr,
                                embed_k=self._factors, l_w=self._l_w, random_seed=self._seed, n_rules=self._n_rules, nint=self._nint,
                                epsilon=self._eps, l_rc=self._l_rc)

    @property
    def name(self):
        return "RBRSINT2" \
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
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            # predictions = self._model.vectorized_predict(offset, offset_stop)
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

        return False
