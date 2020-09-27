import os
import sys
from datetime import datetime
import pickle
import abc
import multiprocessing
from multiprocessing import Pool
from itertools import product
from pprint import pprint

def print_log(*args):
    print("[{}]".format(datetime.now()), *args)
    sys.stdout.flush()

# example
def train_and_eval(param):
    """Accept a specific parameter setting, and run the training and evaluation
    procedure. 

    The training and evaluation procedure are specified by users.
    
    The evaluation procedure should return a `scalar` value to indicate the 
    performance of this parameter setting.

    Args:
    param -- A dict object specifying the parameters.

    Returns:
    A scalar value of evaluation.
    """
    pass


class HyperparameterSearch:
    """Multiprocessing hyperparameter search. It also enables multiple GPU
    usage.

    One worker process is assigned to one GPU. Worker-GPU mapping can be 
    many-to-many.
    """
    def __init__(self, num_worker=1, first_gpu=0, num_gpu=1):
        """
        Args:
        num_worker -- The number of subprocesses in use. An integer larger than 
            one.
        first_gpu -- The id of the first gpu in a consequtive sequence of gpus 
            in use.
        num_gpu -- The number of a consequtive sequence of gpus 
            in use. Specify 0 to indicate not using gpus.
        """
        assert num_worker >= 1 and isinstance(num_worker, int)
        self.num_worker = num_worker
        assert num_gpu >= 0 and isinstance(num_gpu, int)
        self.num_gpu = num_gpu
        if self.num_gpu > 0:
            assert first_gpu >= 0 and isinstance(first_gpu, int) # gpu id
            self.first_gpu = first_gpu
        print_log(
            f"{self.num_worker} worker processes. "
            f"{self.num_gpu} GPUs in use.")

    def _worker(self, param):
        """Assocaite one gpu to a worker if possible.
        """
        if self.num_gpu > 0:
            # set visible gpus
            proc = multiprocessing.current_process()
            proc_id = int(proc.name.split('-')[-1])
            # each process assumes one specific gpu
            gpu_id = self.first_gpu + proc_id % self.num_gpu 
            os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu_id
            print_log(f"GPU-{gpu_id} contains:")
            pprint(param)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if hasattr(self, "train_and_eval"):
            eval_res = self.train_and_eval(param)
            assert isinstance(eval_res, float)
        else:
            raise NameError(
                "train_and_eval() is not specified. "
                "Run hyperparameter_optimization() to add in.")
        return eval_res

    def hyperparameter_optimization(self, train_and_eval, params):
        """Perform hyperparameter searching in the space defined by `params`. 
        
        Args:
        train_and_eval -- A function taking a specific parameter setting, and 
            run the training and evaluation procedure. 

            The training and evaluation procedure are specified by users.
            
            The evaluation procedure should return a `scalar` value to indicate 
            the performance of this parameter setting.

            Args:
            param -- A dict object specifying the parameters.

            Returns:
            A scalar value of evaluation.

        params -- A dict object defining the parameter space to search the
            optimal value. Each item must be iteratable and has the format:
                {
                 'param_1': [N, ...], 
                 'param_2': [N, ...],
                 ...
                }

        Returns:
        A list of dict object that contains the best configuration (note that 
            a repetition of the same values might happen);
        A list of dict object that contains the parameter setting;
        A list of evluation result that corresponds to the parameter setting.
        """
        self.train_and_eval = train_and_eval # add for later usage
        print_log("Parameter settings:")
        pprint(params, width=40)
        possible_combs = product(*params.values())
        # this format is required for multiprocessing
        params_list = [
            dict(zip(params.keys(), c))
                for c in possible_combs]

        with Pool(self.num_worker) as p:
            res_list = p.map(self._worker, params_list)
            p.close()
        
        best_eval_res = max(res_list)
        best_args = [
            i for (i, res) in enumerate(res_list) 
                if best_eval_res == res]
        best_param_list = [params_list[i] for i in best_args]
        # summary
        print_log(
            f"Best parameter setting with evaluation result {best_eval_res}:")
        pprint(best_param_list, width=40)
        print_log("\nOther settings")
        for p, r in zip(params_list, res_list):
            print('-'*78)
            pprint(p, width=80)
            print(f"Evaluation result: {r}")
        return best_param_list, params_list, res_list


if __name__ == '__main__':
    params = {
        "lr": [0.001, 0.01, 0.1],
        "dim": [20, 50, 100],
    }

    def train_and_eval(param):
        return param["lr"]

    hs = HyperparameterSearch(num_worker=1, first_gpu=0, num_gpu=8)
    best_param_list, params_list, res_list = hs.hyperparameter_optimization(
        train_and_eval, params)
    