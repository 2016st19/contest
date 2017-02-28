/**
 * Created by jyb on 2/23/17.
 */

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

public class PreProcess {
    public static class PreProcessMapper
            extends Mapper<Object, Text, Text, Text>{
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String filename = String.valueOf(fileSplit.getPath().getName()).split("_")[0];
            StringTokenizer itr = new StringTokenizer(value.toString());
            if(!((filename.equals("bank")) || (filename.equals("usersID")) || (filename.equals("overdue")))){
                while(itr.hasMoreTokens()){
                    String line = itr.nextToken().toString();
                    context.write(new Text(line.split(",")[0]),
                            new Text(filename + "#" + line.substring(line.indexOf(",") + 1, line.length())));
                }
            }
        }
    }

    public static class PreProcessNewPartitioner
            extends HashPartitioner<Text, Text>{
        public int getPartition(Text key, Text value, int numReduceTasks){
            return super.getPartition(key, value, numReduceTasks);
        }
    }

    public static class PreProcessReducer
            extends Reducer<Text, Text, Text, Text>{
        private Text word = new Text();
        static Text CurrentItem = new Text("*");
        private static int attrNum = 16;
        private static double[] attrDouble = new double[attrNum];
        private static int[] attrInt = new int[attrNum];
        private static String timeStamp;

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException{
            word.set(key);
            if((!CurrentItem.equals(word)) && (!CurrentItem.equals("*"))) {
                myOutPut(context);
                Initial();
            }
            CurrentItem = new Text(word);
            for(Text v : values) {
                String filename = v.toString().split("#")[0];
                if((filename.equals("user")) || (filename.equals("loan"))) {
                    context.write(key, v);
                } else {
                    String[] items = v.toString().split("#")[1].split(",");
                    if(filename.equals("bill")) {
                        for(int i = 0; i <= 13; ++i){
                            if(i == 0) {
                                double buf = Double.parseDouble(items[i]);
                                if(buf > attrDouble[i]){
                                    attrDouble[i] = buf;
                                    timeStamp = items[i];
                                }
                            } else if((i == 1) || (i == 7) || (i == 13)) {
                                int buf = Integer.parseInt(items[i]);
                                attrInt[i] = (buf >= attrInt[i]) ? buf : attrInt[i];
                            } else {
                                double buf = Double.parseDouble(items[i]);
                                attrDouble[i] = (buf >= attrDouble[i]) ? buf : attrDouble[i];
                            }
                        }
                    } else if(filename.equals("browse")) {
                        attrInt[14] += Integer.parseInt(items[1]);
                        attrInt[15] += Integer.parseInt(items[2]);
                    }
                }
            }
        }

        protected static void Initial() {
            for(int i = 0; i < attrNum; ++i) {
                attrDouble[i] = 0;
                attrInt[i] = 0;
            }
            timeStamp = "";
        }

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Initial();
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException{
            myOutPut(context);
        }

        private static void myOutPut(Context reducerContext) throws IOException, InterruptedException {
            String output1 = "";
            for(int i = 0; i <= 13; ++i) {
                if(i == 13) output1 += attrInt[i];
                else {
                    if(i == 0) output1 += timeStamp + ",";
                    else if((i == 1) || (i == 7)) output1 += attrInt[i] + ",";
                    else output1 += attrDouble[i] + ",";
                }
            }
            reducerContext.write(CurrentItem, new Text("bill#" + output1));
            reducerContext.write(CurrentItem, new Text("browse#" + attrInt[14] + "," + attrInt[15]));
        }
    }

    public static void main(String inputPath, String outBufPath) throws Exception{
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "PreProcess");
        job.setNumReduceTasks(30);
        job.setJarByClass(PreProcess.class);
        job.setMapperClass(PreProcessMapper.class);
        job.setPartitionerClass(PreProcessNewPartitioner.class);
        job.setReducerClass(PreProcessReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outBufPath));
        int exit = job.waitForCompletion(true) ? 0 : 1;
    }
}
