uint16_t OSReadSwapInt16(uint16_t value) {
    // Swap the bytes using bitwise operations
    return (value >> 8) | (value << 8);
}