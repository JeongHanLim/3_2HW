#include <unistd.h>
int main()
{
    int x = 56;
    char c = x;
    write(1, &c, sizeof(c));


}