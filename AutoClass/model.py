from .config import default_config_dict
import numpy as np
import random
from collections import Counter
from river import base, metrics, ensemble
from scipy.stats import truncnorm

class AutoClass(base.Classifier):
    def __init__(
        self,
        config_dict=None,
        metric=metrics.Accuracy(),
        exploration_window=1000,
        population_size=10,
        seed=42):
    
        # Configuration
        self.metric = metric
        self.exploration_window = exploration_window
        self.population_size = population_size
        self.config_dict = config_dict
        self._COUNTER = 0
        
        # Set random seeds for reproducibility
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        if config_dict:
            # Get algorithms and hyperparameters
            self._algorithms = config_dict['algorithms']
            self._hyperparameters = config_dict['hyperparameters']
        else:
            self._algorithms = default_config_dict['algorithms']
            self._hyperparameters = default_config_dict['hyperparameters']
        
        # Initialize population
        self._pipeline_list = [self._initialize_random_pipeline(random_hyper=True) for i in range(self.population_size)]
        self._metrics = [type(self.metric)() for _ in range(len(self._pipeline_list))]
        
        # Index of the best pipeline
        self._best_model_idx = 0#np.random.randint(len(self._pipeline_list))
        
        self._max_params = self._max_parameters_length(self._hyperparameters)
        
        # Regression model
        self._regressor = ensemble.AdaptiveRandomForestRegressor()
    
    
    def _max_parameters_length(self, hyperparameters_options):
        # Initialize variable to keep track of the maximum number of parameters
        max_params = 0

        # Iterate through each classifier in the dictionary
        for classifier, params in hyperparameters_options.items():
            # Count the number of parameters for the current classifier
            params_length = len(params)

            # Update max_params if the current classifier has more parameters
            if params_length > max_params:
                max_params = params_length

        return max_params
    
    def _list_to_dict(self, input_list):
        # Initialize an empty dictionary
        result_dict = {}
        # Iterate over the list up to max_params
        for i in range(self._max_params):
            # Assign each value to a key in the dictionary
            # Key is i+1 to start counting from 1
            result_dict[str(i+1)] = input_list[i]
        return result_dict
    
    def _encode_param(self,  p, space):
        if isinstance(p, bool):  # For boolean, encode True as 1 and False as 2
            return 1 if p else 2
        elif isinstance(p, (int, float)):  # For int and float, return as-is
            return p
        elif p in space:  # For categorical, return its 1-based index
            return space.index(p) + 1
        else:  # Default case
            return 1
    
    def _create_input_vector(self, model):
        
        model_name = type(model).__name__
        hyperparameters = self._get_current_params(model)
        
        # Initialize the vector with zeros
        x = [0] * self._max_params
        # Get the space definitions for the model
        param_spaces = self._hyperparameters.get(model_name, {})
        # Iterate through the hyperparameters to encode
        for i, (key, value) in enumerate(hyperparameters.items()):
            if i >= self._max_params:
                break  # Avoid exceeding the max_params limit
            space = param_spaces.get(key, [])  # Get the space for the current parameter
            x[i] = self._encode_param(value, space)
        return self._list_to_dict(x)
                
    
    def _get_hyperparameters(self, model, hyper_init="random"):
        model_name = type(model).__name__
        model_hyper = self._hyperparameters[model_name]

        hyperparser = {}

        if hyper_init == "random":
            for key, values in model_hyper.items():
                hyperparser[key] = random.choice(values)
        else:
            default_hyper = model._get_params()
            for key, values in model_hyper.items():
                if default_hyper[key] in values:
                    hyperparser[key] = default_hyper[key]
                else:
                    hyperparser[key] = values[0]

        return model._set_params(hyperparser)
    
    def _initialize_random_pipeline(self,random_hyper=False):
        """Initialize a random pipeline"""
        
        if random_hyper==True:
            # Ensure that each component type is selected at least once
            random_algorithm = self._get_hyperparameters(random.choice(self._algorithms), hyper_init="random")
            return random_algorithm
        
        return random.choice(self._algorithms)
    
    def _truncated_normal(self, mean, sd, low, upp):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def _mutate_hyperparameters(self, parent_params, model_name):
        """Mutate hyperparameters of a model"""
        lambda_factor = 0.5  # Decay factor for mutation
        reset_probability = 0.01  # Probability to reset std deviation
        mutated_params = {}
        for param, value in parent_params.items():
            if param in self._hyperparameters[model_name]:
                param_range = self._hyperparameters[model_name][param]
                if isinstance(value, (int, float)) and not isinstance(value, bool):  # Numerical or Integer
                    mean = value
                    if isinstance(param_range, list):  # Assuming range is provided as a list
                        low, upp = min(param_range), max(param_range)
                    else:
                        low, upp = value - 1, value + 1  # Default range if not specified
                    sd = (upp - low) / 6
                    sd_child = sd * (2 ** (-lambda_factor))
                    mutated_value = self._truncated_normal(mean, sd_child, low, upp).rvs()
                    if isinstance(value, int):
                        mutated_value = round(mutated_value)
                    mutated_params[param] = mutated_value
                elif isinstance(value, str):  # Categorical
                    # Assuming all categorical options are provided in param_range
                    options = param_range
                    index = options.index(value)
                    probabilities = np.ones(len(options)) / len(options)
                    probabilities[index] *= (2 - 2 ** (-lambda_factor))
                    probabilities /= probabilities.sum()  # Normalize
                    mutated_params[param] = np.random.choice(options, p=probabilities)
                elif isinstance(value, bool):  # Boolean
                    # Assuming a 50-50 chance to flip the boolean value
                    mutated_params[param] = not value if np.random.rand() < 0.5 else value
                # Add logic for Ordinal parameters
                elif isinstance(value, ord):  # Ordinal
                    index = param_range.index(value)
                    upper = len(param_range) - 1
                    if np.random.rand() < reset_probability:
                        std_dev = upper / 2
                    else:
                        std_dev = ((upper - 0) / 6) * np.power(2, -lambda_factor)
                    new_index = int(round(self._truncated_normal(index, std_dev, 0, upper).rvs()))
                    new_index = max(0, min(new_index, upper))
                    mutated_params[param] = param_range[new_index]
                else:
                    mutated_params[param] = value  # Copy parameter if it doesn't need mutation
            else:
                mutated_params[param] = value  # Copy parameter if it's not in the hyperparameters dict
        return mutated_params
    
    def _get_current_params(self, model):
        """Get current hyperparameters of a model"""
        model_name = type(model).__name__
        model_hyper = self._hyperparameters[model_name]
        current_hyper = {}
        for k, _ in model_hyper.items():
            current_hyper[k] = model._get_params()[k]
        return current_hyper

    def _mutate(self, model):
        """Mutate a model pipeline"""
        parent_params = self._get_current_params(model)#model._get_params()
        model_name = type(model).__name__
        #print(parent_params)
        mutated_params = self._mutate_hyperparameters(parent_params, model_name)
        #print(mutated_params)
        # Assuming model can be re-initialized with mutated parameters
        return model._set_params(mutated_params)
    
    def _select_parent(self):
        """Select parent model for mutation"""
        scores = [m.get() for m in self._metrics]
        total = sum(scores)
        probs = [s / total for s in scores]  
        return random.choices(self._pipeline_list, probs)[0]
    
    def _learn_reg(self):
        """Update regression model"""
        for idx, model in enumerate(self._pipeline_list):
            x = self._create_input_vector(model)
            y = round(self._metrics[idx].get()*100,2)
            self._regressor.learn_one(x,y)
 
    def predict_one(self, x):
        """Make prediction for one sample"""
        return self._pipeline_list[self._best_model_idx].predict_one(x)

    def learn_one(self, x, y):
        """Update pipelines with one sample"""
        # Update and train the best model and pipeline list
        for idx, model in enumerate(self._pipeline_list):
            
            try:
                y_pred = model.predict_one(x)
                self._metrics[idx].update(y, y_pred)
                for _ in range(np.random.poisson(6)):
                    model.learn_one(x, y)

                # Check for a new best model
                if self._metrics[idx].is_better_than(self._metrics[self._best_model_idx]):
                    self._best_model_idx = idx
            
            except Exception as e:
                pass
                # Optionally handle exceptions here

        self._COUNTER += 1
        self._check_exploration_phase()
            

    def _check_exploration_phase(self):
        """Check if need to explore"""
        if self._COUNTER % self.exploration_window == 0:
            self._learn_reg()
            parent_model = self._select_parent()
            
            mutate_model = self._mutate(parent_model)
            
            score = self._regressor.predict_one(self._create_input_vector(mutate_model))
            
            worst_model_idx = np.argmin([m.get() for m in self._metrics])
            
            if score>=self._metrics[worst_model_idx].get():
                self._pipeline_list.pop(worst_model_idx)
                self._metrics.pop(worst_model_idx)
                
                self._pipeline_list.append(mutate_model)
                self._metrics.append(type(self.metric)())       