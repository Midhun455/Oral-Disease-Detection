import webbrowser
from django.conf import settings
from django.shortcuts import render, redirect, HttpResponse

from OralProject.settings import BASE_DIR
from .models import *
from pathlib import Path
import uuid
import subprocess
import os

# Create your views here.


def index(request):
    return render(request, "index.html")


def contact(request):
    return render(request, "contact.html")


def chatbot(request):
    # abc = subprocess.run(["python", "gradio_ui.py"], check=True)
    # print("HELLO",abc)
    # import time
    # time.sleep(10)
    webbrowser.open("https://huggingface.co/spaces/ForPythonJava/ChatBot")
    return redirect("/")


def run_yolov5_detection(image_path, output_dir):
    current_directory = os.getcwd()
    detect_script = current_directory + r"\yolov5\detect.py"

    weights_path = current_directory + r"\yolov5\runs\train\exp2\weights\best.pt"

    command = [
        "python",
        detect_script,
        "--weights",
        weights_path,
        "--img",
        "640",  # Set the image size
        "--conf",
        "0.25",  # Set the confidence threshold
        "--source",
        str(image_path),  # Input file
        "--project",
        str(output_dir),  # Output directory
        "--name",
        "detection_results",  # Folder name within the output directory
        "--exist-ok",  # Allow existing files to be overwritten
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    # print("stdout:", result.stdout)
    # print("stderr:", result.stderr)
    # Check for errors
    if result.returncode == 0:
        print("Detection completed successfully.")
        # Process the output as needed
    else:
        print(f"Error during detection: {result.stderr}")
    return result


# python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source data/images


def detect(request):
    if request.POST:
        image = request.FILES["imgfile"]
        unique_filename = str(uuid.uuid4()) + Path(image.name).suffix
        image_path = Path(settings.MEDIA_ROOT) / unique_filename
        print(image_path)
        with open(image_path, "wb+") as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        output_dir = Path(settings.STATICFILES_DIRS[0])
        output_dir.mkdir(parents=True, exist_ok=True)
        # Call the YOLOv5 detection function
        run_yolov5_detection(image_path, output_dir)

        detected_image_path = output_dir / "detection_results" / unique_filename
        relative_detected_image_path = detected_image_path.relative_to(
            Path(settings.STATICFILES_DIRS[0])
        ).as_posix()
        detected_image_url = request.build_absolute_uri(
            f"{settings.STATIC_URL}{relative_detected_image_path}"
        )
        return render(request, "results.html", {"image": detected_image_url})
        # return HttpResponse(
        #     f"Image uploaded and detection performed. <a href='{detected_image_url}'>View result</a>"
        # )
    return render(request, "detect.html")
