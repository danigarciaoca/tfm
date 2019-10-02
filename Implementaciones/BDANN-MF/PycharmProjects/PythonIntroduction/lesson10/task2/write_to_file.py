zoo = ['lion', "elephant", 'monkey']

if __name__ == "__main__":
    f = open("output.txt", "a")

    for i in zoo:
        f.writelines(i)
        f.writelines(" ")

    f.close()
