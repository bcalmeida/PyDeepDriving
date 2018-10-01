import drive
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np

# Test 1
# drive.test_shared_memory()


# Test 2
# drive.setup_shared_memory()
# while True:
#     written = drive.is_written()
#     print("written is " + str(written))
#     inp = input()
#     if inp == 'x':
#         drive.close_shared_memory()
#         break
#     elif inp == 'w':
#         drive.write(not written)


# # Test 3
# @contextmanager
# def shared_memory(*args, **kwds):
#     drive.setup_shared_memory()
#     try:
#         yield None
#     finally:
#         drive.close_shared_memory()

# with shared_memory() as _:
#     while True:
#         written = drive.is_written()
#         print("written is " + str(written))
#         inp = input()
#         if inp == 'x':
#             break
#         elif inp == 'w':
#             drive.write(not written)


# Test 4
# with shared_memory() as _:
#     drive.setup_opencv()
#     while True:
#         paused = drive.is_paused()
#         written = drive.is_written()
#         print("written is ", written)
#         print("paused is ", paused)
#         inp = input("Enter input:")
#         if inp == 'p':
#             drive.pause(not paused)
#         elif inp == 'w':
#             drive.write(not written)


# # Test 5
# WIDTH = 280
# HEIGHT = 210
# with shared_memory() as _:
#     drive.setup_opencv()
#     while True:
#         paused = drive.is_paused()
#         print("paused is ", paused)
#         inp = input("Enter input:")
#         if inp == 'p':
#             drive.pause(not paused)
#         if drive.is_written():
#             print("Reading img")
#             img = drive.read_image() # HWC, BGR
#             img_np = np.asarray(img).reshape(HEIGHT, WIDTH, 3)[:,:,::-1].astype('uint8') # HWC, RGB
#             # import pdb; pdb.set_trace()
#             plt.imshow(img_np)
#             plt.show()
#             drive.write(False) # needs to do this so torcs continues and writes again


# # Test 6: read_indicators
# with shared_memory() as _:
#     drive.setup_opencv()
#     print("paused is ", drive.is_paused())
#     drive.pause(False)
#     print("paused is ", drive.is_paused())
#     while True:
#         if drive.is_written():
#             ground_truth = drive.read_indicators()
#             print(ground_truth)
#             drive.write(False)


# # Test 7: controller
# with shared_memory() as _:
#     drive.setup_opencv()
#     drive.pause(False) # TORCS may share images and ground truth
#     print("Controlling: ", drive.is_controlling())
#     drive.set_control(True)
#     print("Controlling: ", drive.is_controlling())
#     input("Press key to start...")
#     while True:
#         if drive.is_written():
#             ground_truth = drive.read_indicators()
#             print(ground_truth)
#             drive.controller(ground_truth)
#             drive.write(False) # Shared data read, and TORCS may continue


# Test 8: visualization
@contextmanager
def shared_memory(*args, **kwds):
    drive.setup_shared_memory()
    drive.setup_opencv()
    try:
        yield None
    finally:
        drive.close_shared_memory()
        drive.close_opencv()

with shared_memory() as _:
    drive.setup_opencv()
    drive.pause(False) # TORCS may share images and ground truth
    print("Controlling: ", drive.is_controlling())
    drive.set_control(True)
    print("Controlling: ", drive.is_controlling())
    input("Press key to start...")
    while True:
        if drive.is_written():
            ground_truth = drive.read_indicators()
            print(ground_truth)
            drive.controller(ground_truth)
            drive.write(False) # Shared data read, and TORCS may continue
            drive.wait_key(1)
