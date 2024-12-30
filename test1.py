class Human:

    def __init__(self, coloureyes, name, haircolor,):
        self.__eyes = coloureyes
        self.name = name
        self.hair = haircolor

    def get_name(self):
        return self.name
    
    def set_name(self, new_name):
        self.name = new_name

    def get_eyes(self):
        return self.__eyes

    def get_hair_color(self):
        return self.hair
    
    def set_hair_color(self, new_color):
        self.hair = new_color



class Male(Human):
    def __init__(self, eyes, name, haircolor):
        super().__init__(eyes, name, haircolor)
        self.gender ='male'
    def get_eyes(self):
        return 'brown'




human = Human('blue', 'Alex', "blond")

Alex = Male('blue', 'Alex', "blond")

# print(human.__eyes)

print(human.get_eyes())
print(Alex.get_eyes())
# print(Alex.gender)
