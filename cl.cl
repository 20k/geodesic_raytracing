__kernel
void do_raytracing(__write_only image2d_t out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int width = get_image_width(out);

    float frac = (float)x / width;

    write_imagef(out, (int2){x, y}, (float4){frac, 0, 0, 1});
}
