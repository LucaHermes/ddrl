from ray.rllib.models import ModelCatalog
from models.fcnet_glorot_uniform_init import FullyConnectedNetwork_GlorotUniformInitializer
from models.graphnet_glorot_uniform_init import FullyConnectedNetwork_SharedGNN_GlorotUniformInitializer
from models.coupling_net_glorot_uniform_init import FullyConnectedNetwork_Coupling_GlorotUniformInitializer

ModelCatalog.register_custom_model("ffn", FullyConnectedNetwork_GlorotUniformInitializer)
ModelCatalog.register_custom_model("gnn", FullyConnectedNetwork_SharedGNN_GlorotUniformInitializer)
ModelCatalog.register_custom_model("cup", FullyConnectedNetwork_Coupling_GlorotUniformInitializer)

# for backward compatibility
ModelCatalog.register_custom_model("fc_glorot_uniform_init", FullyConnectedNetwork_GlorotUniformInitializer)
