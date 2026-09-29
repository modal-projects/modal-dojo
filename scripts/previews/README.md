The Modal Dojo preview system automatically generates previews of the dashboard and documentation frontends for relevant PRs. When a PR is opened or updated, CI uploads `.tar.gz` artifacts to a Modal volume, then calls a Modal function in `frontend_previews.py` to deploy the preview to a Modal Sandbox. A simple web function, called the redirector, takes care of directing users to the correct sandbox.

For this to work, a few prerequisites need to be met:
* `frontend_previews.py` must be deployed. This is taken care of by the `previews-app.yml` workflow, which is run on every push to `main`, but you may need to manually rerun the workflow if the app was deleted from Modal.
* `PREVIEW_MODAL_TOKEN_ID` and `PREVIEW_MODAL_TOKEN_SECRET` need to be set to a valid Modal API key in the repository's secrets.
* `PREVIEW_BASE_URL` must be set in the repository settings to the URL of the deployed redirector. This is so that CI workflows can post comments with preview URLs.