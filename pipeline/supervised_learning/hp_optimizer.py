import hyperopt

trials = hyperopt.Trials()
best_params = hyperopt.fmin(fn=_objective,
                            space=params,
                            algo=hyperopt.tpe.suggest,
                            max_evals=max_autotune_eval,
                            trials=trials)