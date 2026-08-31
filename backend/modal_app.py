import modal

app = modal.App("vedaai-backend")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("app", remote_path="/root/app")
)


@app.function(
    image=image,
    memory=4096,
    timeout=1800,
    secrets=[modal.Secret.from_name("vedaai-secrets")],
)
@modal.asgi_app()
def fastapi_app():

    from app.main import app as fastapi_app

    return fastapi_app