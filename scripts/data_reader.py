

if __name__ == "__main__":
    import numpy
    import pickle
    import lzma


    with lzma.open("record_3.npz", "rb") as file:
        data = pickle.load(file)

        print("Read", len(data), "snapshotwas")
        print(data[0].image)
        print([("car_speed:",e.car_speed,",raycast_distances:",e.raycast_distances,",current_controls:",e.current_controls) for e in data])
