function data = CreateCircularBuffer(BUFF_DIM, numRow_numCol)
data.index = 1;
data.maxIndex = BUFF_DIM;
if size(numRow_numCol,2) == 2
    data.buffer = zeros(numRow_numCol(1), numRow_numCol(2), BUFF_DIM);
else
    data.buffer = zeros(numRow_numCol, BUFF_DIM);
end
end

