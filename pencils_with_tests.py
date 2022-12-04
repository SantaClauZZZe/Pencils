MIN_THREADS = 1
MAX_THREADS = 11
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

    # print(f"Количество объектов: {len(props)}")

    # print("-----------")

    # for area in sorted([prop.area for prop in props]):
    #     print(area)

    # print("-----------")

    # for prop in props:
    #     print(f"Area -> {prop.area} Eccentricity -> {prop.eccentricity}")

    for prop in props[:]:
        if prop.area < 250000 or prop.area > 500000 or prop.eccentricity < 0.951:
            props.remove(prop)

    # print(f"Количество карандашей: {len(props)}")

    # plt.imshow(tresh)
    # plt.show()
    # cv.namedWindow("Image", cv.WINDOW_GUI_NORMAL)
    # cv.imshow("Image", image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return len(props)

def process_thread(q_task, q_res):
    while True:
        path = q_task.get()

        if isinstance(path, int):
            break

        q_res.put((path, process_image(path)))

        sleep(0.01)

# ----------- main code -----------

best_thread = None
best_time = 100

for i in range(MIN_THREADS, MAX_THREADS + 1):  # for test

    Q_THREADS = i  # for test

    print(f"Количество потоков: {Q_THREADS}")

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

    if timer < best_time:
        best_time = timer
        best_thread = Q_THREADS

    print(f"Elapsed: {timer}s\n\n-------------------")

print(f"Best results: time -> {best_time} ; threads -> {best_thread}")