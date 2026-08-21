FROM python:3.13-slim

# Keep Python lean and unbuffered so logs stream to `docker compose logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Requirements first so pip layers cache across source edits.
COPY requirements.txt .

# Docling depends on torch and torchvision. Install BOTH from PyTorch's own
# index: the default PyPI wheels bundle CUDA libraries (~2GB) that are dead
# weight here, since the host GPU is compute capability 3.5 and torch needs
# newer. Its own layer so it caches independently.
#
# Both, not just torch: torchvision ships compiled ops that register against a
# specific torch build. Mixing a CPU-index torch with a PyPI torchvision gives
# "RuntimeError: operator torchvision::nms does not exist" the moment Docling
# loads its layout model.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Docling -> rapidocr pulls the full opencv-python, whose wheel links against
# X11 libs (libxcb, libGL) that python:3.13-slim doesn't ship, so `import cv2`
# dies at Docling's table model. The headless build exposes the same cv2
# module with no GUI dependencies — cheaper than apt-installing an X stack.
RUN pip uninstall -y opencv-python && pip install --no-cache-dir opencv-python-headless

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
