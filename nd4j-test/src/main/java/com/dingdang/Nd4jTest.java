package com.dingdang;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Hello world!
 *
 */
public class Nd4jTest
{
    public static void main( String[] args )
    {
        System.out.println( "Start ND4J Testing!" );

        INDArray nd = Nd4j.create(new float[]{1,2,3,4,5,6},new int[]{3,2}, 'c');
        System.out.println(nd);
    }
}
