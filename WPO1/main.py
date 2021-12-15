# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    NAME = "right"
    if first_run:
        store_click(get_path(NAME + ".jpg"), NAME + ".txt")
    else:
        xr, mr = calibration(NAME)
    NAME = "left"
    if first_run:
        store_click(get_path(NAME + ".jpg"), NAME + ".txt")
    else:
        xl, ml = calibration(NAME)
    if not first_run:
        compute_a(mr, ml, xr, xl)
        print(points_3d)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
