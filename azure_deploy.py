from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

ws = Workspace.from_config()
model = Model.register(workspace=ws, model_path="pricing_model.pkl", model_name="RetailPricingModel")

inference_config = InferenceConfig(entry_script="score.py", environment=env)
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name="retail-pricing-service",
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)
service.wait_for_deployment(show_output=True)
