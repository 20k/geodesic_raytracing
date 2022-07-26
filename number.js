"use strict";
"use math";

class number
{
    constructor(x)
    {
        CShim.construct(this, x);
    }
};

number.prototype[Symbol.operatorSet] = Operators.create(
{
    "+"(p1, p2)
    {
        return CShim.add(p1, p2);
    },
    "*"(p1, p2)
    {
        return CShim.mul(p1, p2);
    },
    "-"(p1, p2)
    {
        return CShim.sub(p1, p2);
    },
    "/"(p1, p2)
    {
        return CShim.div(p1, p2);
    },
    "neg"(p1)
    {
        return CShim.neg(p1);
    }
},
{
    left: Number,
    "+"(p1, p2)
    {
        return CShim.add(p1, p2);
    },
    "*"(p1, p2)
    {
        return CShim.mul(p1, p2);
    },
    "-"(p1, p2)
    {
        return CShim.sub(p1, p2);
    },
    "/"(p1, p2)
    {
        return CShim.div(p1, p2);
    }
},
{
    right: Number,
    "+"(p1, p2)
    {
        return CShim.add(p1, p2);
    },
    "*"(p1, p2)
    {
        return CShim.mul(p1, p2);
    },
    "-"(p1, p2)
    {
        return CShim.sub(p1, p2);
    },
    "/"(p1, p2)
    {
        return CShim.div(p1, p2);
    }
},
);

CMath.number = number;
CMath.i = CMath.get_i();

function make_class(v)
{
    return new number(v);
}
