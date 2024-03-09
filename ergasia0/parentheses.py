
#1115202000128 ΓΕΩΡΓΙΟΣ ΜΗΤΡΟΠΟΥΛΟΣ

import sys

class Stack:
    def __init__(self):
        self.items = []

    def empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.empty():
            return self.items.pop()
        else:
            return None
        
def is_balanced(expression):
    stack = Stack()
    opening_brackets = "([{"
    closing_brackets = ")]}"

    for bracket in expression:  #for every character in the expression
        if bracket in opening_brackets: #if its open bracket push in stack
            stack.push(bracket)
        elif bracket in closing_brackets: #if is close bracket check if the stack is empty
            if stack.empty():
                return False
            top = stack.pop()
            if opening_brackets.index(top) != closing_brackets.index(bracket): #if it is not, pop the top bracket and check the if the element that poped out is matching with the closing bracket
                return False

    return stack.empty()
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("wrong argument")
    else:
        input = sys.argv[1]
        if is_balanced(input):
            print(f"{input} is balanced.")
        else:
            print(f"{input} is not balanced.")