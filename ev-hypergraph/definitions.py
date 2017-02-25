from enum import Enum

class Category(Enum):
    empty = 0
    exhigh=1
    vhigh =2
    high =3
    mhigh = 4
    mlow=5
    low=6
    vlow=7
    exlow=8
    boolean = 9
class Attribute(Enum):
    capacity = 0
    reliability = 1
    discharge = 2
    commit = 3

gaussianAttributes = {Attribute.capacity : (100, 80), Attribute.reliability : (0, 1), Attribute.discharge : (10, 5)}
booleanAttributes = {Attribute.commit: 0.9}

def getWeight(category):
    if(category == Category.empty):
        return 0
    if(category == Category.exhigh):
        return 128
    if(category == Category.vhigh):
        return 64
    if(category == Category.high):
        return 32
    if(category == Category.mhigh):
        return 16
    if(category == Category.mlow):
        return 8
    if(category == Category.low):
        return 4
    if(category == Category.vlow):
        return 2
    if(category == Category.exlow):
        return 1
    if(category == Category.boolean):
        return 256

def assignCategory(name, value):
    if (name in booleanAttributes):
        if(value):
            return Category.boolean
    if(name in gaussianAttributes):
        (m, sigma) = gaussianAttributes.get(name)
        if(value > m+3*sigma):
            return Category.exhigh
        if(value > m+2*sigma):
            return Category.vhigh
        if(value > m+1*sigma):
            return Category.high
        if(value > m+0*sigma):
            return Category.mhigh
        if(value > m-1*sigma):
            return Category.mlow
        if(value > m-2*sigma):
            return Category.low
        if(value > m-3*sigma):
            return Category.vlow
        else:
            return Category.exlow
    return Category.empty
