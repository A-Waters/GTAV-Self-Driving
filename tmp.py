# Python code to demonstrate String encoding
 
# initialising a String
a = 'GeeksforGeeks'
 
# initialising a byte object
c = b'GeeksforGeeks'
print(type(c))
 
# using encode() to encode the String
# encoded version of a is stored in d
# using ASCII mapping
d = a.encode('ASCII')
 
# checking if a is converted to bytes or not
if (d==c):
    print ("Encoding successful")
else : print ("Encoding Unsuccessful")