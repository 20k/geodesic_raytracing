#ifndef COMMON_CL_INCLUDED
#define COMMON_CL_INCLUDED

#define CONSTEXPR

#ifdef __cplusplus
#undef CONSTEXPR
#define CONSTEXPR constexpr

constexpr
float min(float v1, float v2)
{
    return std::min(v1, v2);
}

constexpr
float max(float v1, float v2)
{
    return std::max(v1, v2);
}
#endif

int get_chunk_size(int in, int width)
{
    if((in % width) == 0)
        return in/width;

    return (in/width) + 1;
}

///ok. How do we know of the range [0, PI/2], and [16.2pi, 18.3pi] overlap?
///or more generally, [15.1pi, 16.2pi] and [32pi, 34pi] overlap

///step 1: if either range is > period, return true
///step 2: find the min periodic boundary of both, eg if our period is 2 * PI, and we have the range [15.1, 16.2*pi], thats
///14pi, for a range of [14pi, 16pi]
///step 3: shear the range into two components, this gives us in the above example
///two ranges, [15.1pi, 16] and [16, 16.2]. Transform the sheared range into the periodic range, ie
///[16, 16.2] -> [14, 14.2]

///we do this for both periodic coordinates, and end up with two ranges. Check if either of them overlap

///we have guaranteed a smooth coordinate system via a lot of effort, that means that the input coordinates are NOT periodic
///in a periodic coordinate system
CONSTEXPR
bool range_overlaps(float s0, float s1, float e0, float e1)
{
    float ns0 = min(s0, s1);
    float ns1 = max(s0, s1);

    float ne0 = min(e0, e1);
    float ne1 = max(e0, e1);

    return ns0 <= ne1 && ne0 <= ns1;
}

CONSTEXPR
bool periodic_range_overlaps(float s1, float s2, float e1, float e2, float period)
{
    if(s2 - s1 >= period)
        return true;

    if(e2 - e1 >= period)
        return true;

    float period_start1 = floor(s1 / period) * period;
    float period_start2 = floor(e1 / period) * period;

    s1 -= period_start1;
    s2 -= period_start1;

    e1 -= period_start2;
    e2 -= period_start2;

    ///transform both into the same range, we now have eg
    ///[1.1pi, 2.4pi] and [0.1pi, 0.3pi]. Deliberate example used

    ///next up, split off the ends of both

    bool has_end_1 = false;
    float split_end1_s = 0;
    float split_end1_e = 0;

    bool has_end_2 = false;
    float split_end2_s = 0;
    float split_end2_e = 0;

    if(s2 >= period)
    {
        has_end_1 = true;
        split_end1_s = 0; ///the new split end starts at the period, and because we're periodic this is 0
        split_end1_e = s2 - period;
        s2 = period;
    }

    if(e2 >= period)
    {
        has_end_2 = true;
        split_end2_s = 0;
        split_end2_e = e2 - period;
        e2 = period;
    }

    ///we now have possibly 2 pairs of ranges

    if(range_overlaps(s1, s2, e1, e2))
        return true;

    if(has_end_1 && range_overlaps(split_end1_s, split_end1_e, e1, e2))
        return true;

    if(has_end_2 && range_overlaps(s1, s2, split_end2_s, split_end2_e))
        return true;

    if(has_end_1 && has_end_2 && range_overlaps(split_end1_s, split_end1_e, split_end2_s, split_end2_e))
        return true;

    return false;
}
#endif
