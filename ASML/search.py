from .config import default_config_dict
import math, random
import numpy as np



class PipelineSearch:
    
    """
    Initializes the PipelineSearch class.

    Parameters:
    - config_dict: Optional dictionary specifying the configuration for the pipeline search.
    - no_fs_models: List of models that do not require feature selection.
    - budget: The total number of pipelines to evaluate.
    """
    
    def __init__(self, config_dict=None,
                 no_fs_models=[],
                 budget=10):
        
        self.config_dict = config_dict
        self.no_fs_models = no_fs_models
        
        self.pipeline_list = []
        
        self.budget = budget

        # Allocate half of the budget to random pipeline selection
        self.random_pipeline_budget = math.ceil(self.budget/2) #round(0.5 * self.budget)
        self.ardns_pipeline_budegt = self.budget-self.random_pipeline_budget#round(0.5 * self.budget)
        
        # Load configurations from the provided dictionary or the default configuration
        if not config_dict:
            # Extract information from default_config_dict
            self.hyperparameters = default_config_dict.get('hyperparameters', {})
            self.algorithms = [self._get_hyperparameters(algo, hyper_init="default") for algo in default_config_dict.get('models', [])]
            self.preprocessing_steps = [self._get_hyperparameters(pre, hyper_init="default") for pre in default_config_dict.get('preprocessors', [])]
            self.feature_selection_methods = [self._get_hyperparameters(fs, hyper_init="default") for fs in default_config_dict.get('features', [])]
        else:
            # Extract information from config_dict
            self.hyperparameters = config_dict.get('hyperparameters', {})
            self.algorithms = [self._get_hyperparameters(algo, hyper_init="default") for algo in config_dict.get('models', [])]
            self.preprocessing_steps = [self._get_hyperparameters(pre, hyper_init="default") for pre in config_dict.get('preprocessors', [])]
            self.feature_selection_methods = [self._get_hyperparameters(fs, hyper_init="default") for fs in config_dict.get('features', [])]
            
    def _get_hyperparameters(self, model, hyper_init="random"):
        """
        Retrieves hyperparameters for a given model.

        Parameters:
        - model: The model for which to retrieve hyperparameters.
        - hyper_init: The initialization mode for hyperparameters ("random" or "default").
        
        Returns:
        - A dictionary of hyperparameters for the model.
        """
        model_name = type(model).__name__
        model_hyper = self.hyperparameters[model_name]

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
    
    def _create_pipelines(self):
        """
        Creates a list of pipelines based on the available preprocessing steps, models, and feature selection methods.

        Returns:
        - A list of pipelines.
        """
        for preprocessing_step in self.preprocessing_steps:
            for model_instance in self.algorithms:
                # Create a pipeline without feature selection
                pipeline = preprocessing_step | model_instance
                self.pipeline_list.append(pipeline.clone())

                if type(model_instance).__name__ not in self.no_fs_models:
                    # Create a pipeline with feature selection
                    for feature_selection in self.feature_selection_methods:
                        full_pipeline = preprocessing_step | feature_selection | model_instance
                        self.pipeline_list.append(full_pipeline.clone())
        
        return self.pipeline_list
    
    def _initialize_random_pipeline(self,random_hyper=False):
        """
        Initializes a random pipeline.

        Parameters:
        - random_hyper: Whether to randomly initialize hyperparameters.

        Returns:
        - A randomly initialized pipeline.
        """

        # Ensure that each component type is selected at least once
        random_algorithm = random.choice(self.algorithms)
        if len(self.preprocessing_steps)!=0:
            random_preprocessing = random.choice(self.preprocessing_steps)
        if len(self.feature_selection_methods)!=0:
            random_feature_selection = random.choice(self.feature_selection_methods)
        
        if random_hyper:
            random_algorithm = self._get_hyperparameters(random_algorithm, hyper_init="random")
            random_preprocessing = self._get_hyperparameters(random_preprocessing, hyper_init="random")
            random_feature_selection = self._get_hyperparameters(random_feature_selection, hyper_init="random")

        new_choice = random.choice(["WITH_FEATURE", "WITHOUT_FEATURE"])

        if type(random_algorithm).__name__ in self.no_fs_models:
            new_choice = "WITHOUT_FEATURE"

        if new_choice == "WITH_FEATURE":
            random_feature_selection = random.choice(self.feature_selection_methods)
            return random_preprocessing | random_feature_selection | random_algorithm
        else:
            return random_preprocessing | random_algorithm
    
    def _get_current_params(self, model):
        
        """
        Retrieves the current parameters of a given model.

        Parameters:
        - model: The model from which to retrieve parameters.

        Returns:
        - A dictionary of the model's current parameters.
        """

        model_name = type(model).__name__
        model_hyper = self.hyperparameters[model_name]
        current_hyper = {}
        for k, _ in model_hyper.items():
            current_hyper[k] = model._get_params()[k]
        return current_hyper

    def _ardns(self, current_value, values_list):
        """
        Suggests a new value for a hyperparameter based on the ARDNS strategy.

        Parameters:
        - current_value: The current value of the hyperparameter.
        - values_list: The list of possible values for the hyperparameter.

        Returns:
        - A new value for the hyperparameter.
        """

        nearby_option = np.random.choice(["same", "upper", "lower", "random"])

        if nearby_option == "same":
            return current_value
        elif nearby_option == "upper":
            return values_list[min(values_list.index(current_value) + 1, len(values_list) - 1)]
        elif nearby_option == "lower":
            return values_list[max(values_list.index(current_value) - 1, 0)]
        else:  # "random"
            return np.random.choice(values_list)

    def _suggest_nearby_hyperparameters(self, model):
        
        """
        Suggests nearby hyperparameters for a given model based on the ARDNS strategy.

        Parameters:
        - model: The model for which to suggest nearby hyperparameters.

        Returns:
        - A model with suggested hyperparameters.
        """

        # Get the current hyperparameter values
        current_hyperparameters = self._get_current_params(model)

        # Get the user define hyperparameter values
        user_defined_search_space = self.hyperparameters[type(model).__name__]
        # print(user_defined_search_space)

        suggested_hyperparameters = {}

        for param_name, param_value in current_hyperparameters.items():
            if isinstance(param_value, int):
                # Integer hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = int(new_value)

            elif isinstance(param_value, float):
                # Float hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = float(new_value)

            elif isinstance(param_value, (str, bool)):
                # Categorical hyperparameter
                new_value = self._ardns(param_value, user_defined_search_space[param_name])

                suggested_hyperparameters[param_name] = new_value

            else:
                # Unsupported type (e.g., NoneType)
                # random value selected from search space
                suggested_hyperparameters[param_name] = random.choice(user_defined_search_space[param_name])

        return model._set_params(suggested_hyperparameters)
        # return model.clone(new_params=suggested_hyperparameters)
    
    def next_nerby(self, pipeline):
        
        """
        Generates the next nearby pipeline based on the ARDNS strategy.

        Parameters:
        - pipeline: The current pipeline.

        Returns:
        - The next nearby pipeline.
        """

        next_nerby_pipeline = list(pipeline.steps.values())

        new_algorithm = self._suggest_nearby_hyperparameters(next_nerby_pipeline[-1])
        new_preprocessing = self._suggest_nearby_hyperparameters(next_nerby_pipeline[0])

        if len(next_nerby_pipeline) > 2:
            new_feature_selection = self._suggest_nearby_hyperparameters(next_nerby_pipeline[1])
            return new_preprocessing | new_feature_selection | new_algorithm
        else:
            return new_preprocessing | new_algorithm
    
    def select_and_update_pipelines(self, best_pipeline):
        """
        Selects and updates pipelines based on the best performing pipeline.

        Parameters:
        - best_pipeline: The best performing pipeline.

        Returns:
        - A list of updated pipelines.
        """
        
        pipeline = best_pipeline.clone()
        
        next_nerby_pipelines = []
        
        for _ in range(self.ardns_pipeline_budegt):
            next_nerby_pipeline = self.next_nerby(pipeline)
            next_nerby_pipelines.append(next_nerby_pipeline)
            pipeline = next_nerby_pipeline.clone()
        
        new_pipelines = [self._initialize_random_pipeline(random_hyper=False).clone() for i in range(self.random_pipeline_budget)]
        self.pipeline_list = [best_pipeline.clone()] + next_nerby_pipelines + new_pipelines
        return self.pipeline_list