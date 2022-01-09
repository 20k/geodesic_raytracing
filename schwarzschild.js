function metric(t, r, theta, phi)
{
    t = new CMath.number(t);
    r = new CMath.number(r);
    theta = new CMath.number(theta);
    phi = new CMath.number(phi);

    return [-1, 1, r * r, r * r * CMath.sin(theta) * CMath.sin(theta)];
}

metric
