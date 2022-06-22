---
title: "Delete all resources"
weight: 3
---

This workshop creates the following resources:

* SageMaker Endpoints
* S3 objects
* SageMaker apps

If you completed section 2.2, the "Delete resources" section at the end deletes running SageMaker Endpoints and all S3 objects created during the workshop.

You can also delete the endpoints by navigating to AWS Console > Amazon SageMaker. In the left menu click on Inference > Endpoints. Select the endpoint you want to delete and click on Action > Delete.

![](/images/cleanup/cleanup3.png)

For additional information about deleting SageMaker resources, please visit the following documentation page:
https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html

### Deleting SageMaker Studio apps

##### Using the AWS Console
SageMaker Studio also creates apps such as JupyterServer (used to run notebooks), DataWrangler and Debugger which run on EC2 instances. Use the following instructions to shutdown running apps:

Navigate to AWS Console > Amazon SageMaker > Amazon SageMaker Studio. This will open up the SageMaker Studio Control Panel. Click on the Studio user who’s resources you want to delete.

![](/images/cleanup/cleanup1.png)

Under User Details click on “Delete app” to delete all running apps. Keep the “default” App if you want to continue working with SageMaker Studio and want to launch new notebooks.

![](/images/cleanup/cleanup2.png)

##### Using the SageMaker Studio
In SageMaker Studio Notebook, click on the running apps menu which is 3rd from the top. Click on all the power buttons to shut down apps. Keep the running instances if you want to continue working on SageMaker Notebook.

![](/images/cleanup/cleanup4.png)

For more information about deleting Studio resources, Studio domain and how to delete resources using AWS CLI visit the following documentation page:
https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-delete-domain.html?icmpid=docs_sagemaker_console_studio
