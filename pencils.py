Q_THREADS = 6

import os
from pathlib import Path
from threading import Thread
from time import perf_counter, sleep
from queue import Queue

import cv2 as cv
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def process_image(path_iamge) -> int:
    image = cv.imread(path_iamge)
    g_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    _, tresh = cv.threshold(g_image, 130, 255, 1)

    tresh = cv.erode(tresh, None, iterations=8)
    tresh = cv.dilate(tresh, None, iterations=8)

    labeled = label(tresh)
    props = regionprops(labeled)

    for prop in props[:]:
        if prop.area < 250000 or prop.area > 500000 or prop.eccentricity < 0.951:
            props.remove(prop)

    return len(props)

def process_thread(q_task, q_res):
    while True:
        path = q_task.get()

        if isinstance(path, int):
            break

        q_res.put((path, process_image(path)))

        sleep(0.01)

# ----------- main code -----------

print("Preparation...", end="")

path_imgs = Path("images")
files = list(path_imgs.glob("*.jpg"))

if len(files) == 0:
    raise IOError("files not found")

threads = []
task_data = Queue()
res_data = Queue()

timer = perf_counter()

print("\rLaunch threads...", end="")

for i in range(Q_THREADS):
    threads.append(Thread(target=process_thread, args=(task_data, res_data, )))
    threads[-1].start()

print("\rStarting work...", end="")

for name_file in files:
    task_data.put(os.path.dirname(name_file) + os.path.sep + os.path.basename(name_file))

print("\rWait threads work", end="")

for i in range(Q_THREADS):
    task_data.put(-1)

for i in range(Q_THREADS):
    threads[i].join()

timer = perf_counter() - timer

print("\rEnd of work...   ")

print(f"Всего карандашей: {sum( [res_data.get(block=False, timeout=None)[1] for _ in range(res_data.qsize())] )}")

print(f"Elapsed: {timer}s\n\n-------------------")