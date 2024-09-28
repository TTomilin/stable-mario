def converttorgb(hex):
    score = 0
    for char in hex:
        score *= 16
        if char.isnumeric():
            score += int(char)
        else:
            score += int(ord(char))-55
    return score

def converttoehex(integer):
    integer //= 8
    if integer < 10:
        return integer
    return chr(integer + 55)
    
choice = input("1: HEX, 2: RGB\n")
if choice == '1':
    hexval = input("HEX:")
    red = converttorgb(hexval[0:2])
    green = converttorgb(hexval[2:4])
    blue = converttorgb(hexval[4:6])
elif choice == '2':
    red = int(input("Red:"))
    green = int(input("Green:"))
    blue = int(input("Blue:"))

print(converttoehex(red), converttoehex(green), converttoehex(blue), sep='')
