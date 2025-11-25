
def check_values(numbers):
    for n in numbers:   
        if n > 10:
            print("Value is greater than 10:", n)
        elif n == 10:
            print("Value is exactly 10:", n)
        else:
            print("Value is less than 10:", n) 

def process_data(data):
    total = 0
    for item in data:
        if type(item) == int:
            total += item
        if type(item) == str:    
            print("Found string:", item)
        if item == None:     
            print("Found None value")
    return total

values = [5, 10, 15, "hello", None, 3]
result = process_data(values)
print("Total is:", result)

check_values(values)

