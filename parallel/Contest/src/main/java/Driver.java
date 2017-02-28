/**
 * Created by jyb on 2/23/17.
 */
public class Driver {
    public static void main(String[] args) throws Exception{
        String inputPath = args[0];
        String outPath = args[1] + "/buf";
        PreProcess.main(inputPath, outPath);
        inputPath = outPath;
        outPath = args[1] + "/result";
        Join.main(inputPath, outPath);
    }
}
