# M is a square matrix which is already defined.
M = [[1,2,3],
     [2,3,4],
     [3,4,5]]

sym = True
for i in range(len(M)):
    for j in range(i):
        if(M[i][j] != M[j][i]):
            sym = False
            break
    if not sym:
        break            
if sym:
    print('YES')
else:
    print('NO')