
When using jupyter notebook you can use a token to use vc code to access the jupyter noterbook server on localhost:8888
Start the server
- cd notebook
- jupyter notebook

There you can find the token in line 30 which you can use.
Open notebook file in vc code, and click the kernel in the upper right corner and select "Select another kernel"
--> "Existing Jupyter Server..." --> "Enter the URL of the running Jupyter server" and give the whole link with token.



  _   _          _      _
 | | | |_ __  __| |__ _| |_ ___
 | |_| | '_ \/ _` / _` |  _/ -_)
  \___/| .__/\__,_\__,_|\__\___|
       |_|
                       
Read the migration plan to Notebook 7 to learn about the new features and the actions to take if you are using extensions.

https://jupyter-notebook.readthedocs.io/en/latest/migrate_to_notebook7.html

Please note that updating to Notebook 7 might break some of your extensions.

[I 2023-09-03 15:06:41.949 LabApp] JupyterLab extension loaded from /home/peter/anaconda3/lib/python3.11/site-packages/jupyterlab
[I 2023-09-03 15:06:41.949 LabApp] JupyterLab application directory is /home/peter/anaconda3/share/jupyter/lab
NumExpr defaulting to 4 threads.
[I 15:06:48.905 NotebookApp] Serving notebooks from local directory: /home/peter/my-ml-zoomcamp/01-intro/notebooks
[I 15:06:48.905 NotebookApp] Jupyter Notebook 6.5.4 is running at:
[I 15:06:48.905 NotebookApp] http://localhost:8888/?token=c9cf09631f959150469205e586563a3dd5ec90d0a70acfe1
[I 15:06:48.905 NotebookApp]  or http://127.0.0.1:8888/?token=c9cf09631f959150469205e586563a3dd5ec90d0a70acfe1
[I 15:06:48.906 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).