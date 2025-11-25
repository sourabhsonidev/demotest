# Example Python file with intentional issues (style violations, logic mistakes, indentation problems)

def check_values(numbers):
for n in numbers:   # Indentation issue here
    if n > 10:
        print("Value is greater than 10:", n)
    elif n == 10:
        print("Value is exactly 10:", n)
    else:
      print("Value is less than 10:", n) # inconsistent indentation

def process_data(data):
    total = 0
    for item in data:
        if type(item) == int:
            total += item
        if type(item) == str:     # Should probably be elif
            print("Found string:", item)
        if item == None:          # None comparison issue
            print("Found None value")
    return total

values = [5, 10, 15, "hello", None, 3]
result = process_data(values)
print("Total is:", result)

check_values(values)

