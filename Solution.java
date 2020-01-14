package LeetCode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;

class Solution {
	public static void main(String[] args) {
		int i=0;
		System.out.println(i);
		int[] arr=new int[3];

	}
	
    
	public List<List<Integer>> downUpView(TreeNode root) {
        class Element{
            TreeNode tn;
            int index;
            Element(TreeNode tn, int index){
                this.tn=tn;
                this.index=index;
            }
        }
        List<List<Integer>> res=new ArrayList<>();
        if(root==null)return res;
        Queue<Element> queue=new LinkedList<>();
        queue.add(new Element(root, 0));
        Map<Integer, List<Integer>> map=new TreeMap<>();
        while(!queue.isEmpty()){
        	int size=queue.size();
        	Map<Integer, List<Integer>> curLevel=new HashMap<>();
        	while(size-->0) {
	            Element cur=queue.poll();
	            TreeNode curTn=cur.tn;
	            int curIndex=cur.index;
	            if(curLevel.containsKey(curIndex)) {
	            	curLevel.get(curIndex).add(curTn.val);
	            }else {
	            	List<Integer> list=new ArrayList<>();
	            	list.add(curTn.val);
	            	curLevel.put(curIndex, list);
	            }
	            if(curTn.left!=null)queue.offer(new Element(curTn.left, curIndex-1));
	            if(curTn.right!=null)queue.offer(new Element(curTn.right, curIndex+1));
        	}
        	for(int i: curLevel.keySet()) {
        		map.put(i,curLevel.get(i));
        	}
        }
        for(Integer i: map.keySet()){
            res.add(map.get(i));
        }
        return res;
    }
	
	public void helper(int[][] matrix, int index, List<Integer> res){
        int row=matrix.length-index-1;
        int col=matrix[0].length-index-1;
        if(index>row||index>col)return;
        if(index==row){
            for(int j=index;j<=col;j++){
                res.add(matrix[index][j]);
            }
            return;
        }
        if(index==col){
            for(int i=index;i<=row;i++){
                res.add(matrix[i][index]);
            }
            return;
        }
        for(int j=index;j<col;j++){
            res.add(matrix[index][j]);
        }
        for(int i=index;i<row;i++){
            res.add(matrix[i][col]);
        }
        for(int j=col;j>index;j--){
            res.add(matrix[row][j]);
        }
        for(int i=row;i>index;i--){
            res.add(matrix[i][index]);
        }
        helper(matrix, index+1, res);
    }
	
	 public static List<Integer> spiralOrder2(int[][] matrix) {
	     List<Integer> list=new ArrayList<>();
	    if(matrix.length==0) {
	    	return list;
	    }
	    int rowBegin=0;
	    int rowEnd=matrix.length-1;
	    int colBegin=0;
	    int colEnd=matrix[0].length-1;
	    int count=0;
	    int nums=matrix.length*matrix[0].length;
	    while(rowBegin<=rowEnd&&colBegin<=colEnd) {
	    	for(int j=colBegin;j<=colEnd;j++) {
	    		list.add(matrix[rowBegin][j]);
	    	}
	    	rowBegin++;
	    	for(int i=rowBegin;i<=rowEnd;i++) {
	    		list.add(matrix[i][colEnd]);
	    	}
	    	colEnd--;
	    	if(rowBegin<=rowEnd&&colBegin<=colEnd) {
	    		for(int j=colEnd;j>=colBegin;j--) {
	        		list.add(matrix[rowEnd][j]);
	        	}
	    	}
	    	rowEnd--;
	    	if(rowBegin<=rowEnd&&colBegin<=colEnd) {
	    		for(int i=rowEnd;i>=rowBegin;i--) {
	        		list.add(matrix[i][colBegin]);
	        	}
	    	}
	    	colBegin++;
	    }
	    return list;
	}
	 
	public static int diff(int[] nums) {
		int len=nums.length;
		int sum=0;
		for(int m: nums) {
			sum+=m;
		}
		int targetSum=sum/2;
		int[][] dp=new int[len+1][targetSum+1];
		for(int i=1;i<=len;i++) {
			for(int j=1; j<=targetSum; j++) {
				if(j>=nums[i-1]) {
					dp[i][j]=Math.max(dp[i-1][j], dp[i-1][j-nums[i-1]]+nums[i-1]);
				}else {
					dp[i][j]=dp[i-1][j];
				}
			}
		}
		return sum-2*dp[len][targetSum];
	}
		
	public static int longestConsecutive(int[] nums) {
        if(nums==null||nums.length==0)return 0;
        Map<Integer, Integer> map=new HashMap<>();
        int len=nums.length;
        int res=0;
        for(int i=0;i<len;i++){
            if(!map.containsKey(nums[i])){
                int left=map.containsKey(nums[i]-1)?map.get(nums[i]-1):0;
                int right=map.containsKey(nums[i]+1)?map.get(nums[i]+1):0;
                int sum=left+right+1;
                res=Math.max(res, sum);
                map.put(nums[i]-left, sum);
                map.put(nums[i]+right, sum);
            }
        }  
        return res;
    }
	
	public boolean isValidSudoku(char[][] board) {
        int n=board.length;
        Set<Character>[] row=new HashSet[n];
        Set<Character>[] col=new HashSet[n];
        Set<Character>[] small=new HashSet[n];
        for(int i=0;i<n;i++){
            row[i]=new HashSet<>();
            col[i]=new HashSet<>();
            small[i]=new HashSet<>();
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(board[i][j]=='.')continue;
                int inX=i/3, inY=j/3*3%3;
                int index=inX+inY;
                if(row[i].contains(board[i][j])||col[j].contains(board[i][j])||small[index].contains(board[i][j])) return false;
                   row[i].add(board[i][j]);
                   col[j].add(board[i][j]);
                   small[index].add(board[i][j]);
            }
        }
        return true;
    }
	
	public static List<Integer> mergeKSortedList(List<List<Integer>> inputList){
		return helper(inputList, 0, inputList.size()-1);
	}
	
	public static List<Integer> helper(List<List<Integer>> inputList, int left, int right){
		if(left==right)return inputList.get(left);
		int mid=left+(right-left)/2;
		List<Integer> leftSortedList=helper(inputList, left, mid);
		List<Integer> rightSortedList=helper(inputList, mid+1, right);
		return merge(leftSortedList, rightSortedList);
	}
	
	public static List<Integer> merge(List<Integer> left, List<Integer> right){
		int leftSize=left.size();
		int rightSize=right.size();
		int i=0, j=0;
		List<Integer> res=new ArrayList<>();
		while(i<leftSize&&j<rightSize) {
			if(left.get(i)<right.get(j)) {
				res.add(left.get(i++));
			}else {
				res.add(right.get(j++));
			}
		}
		while(i<leftSize) {
			res.add(left.get(i++));
		}
		while(j<rightSize) {
			res.add(right.get(j++));
		}
		return res;
	}
	
	public List<Integer> closestKValues(TreeNode root, double target, int k) {
        List<Integer> res=new ArrayList<>();
        Stack<TreeNode> left=new Stack<>();
        Stack<TreeNode> right=new Stack<>();
        TreeNode cur=root;
        while(cur!=null){
            if(cur.val<target){
                left.push(cur);
                cur=cur.right;
            }else{
                right.push(cur);
                cur=cur.left;
            }
        }
        while(k-->0){
            if(!left.isEmpty()&&!right.isEmpty()){
                if((target-(double)left.peek().val)<((double)right.peek().val-target)){
                    res.add(getNextLeft(left));
                }else{
                    res.add(getNextRight(right));
                }
            }else if(!left.isEmpty()){
                res.add(getNextLeft(left));
            }else{
                res.add(getNextRight(right));
            }
        }
        return res;
    }
    
    public int getNextLeft(Stack<TreeNode> left){
        TreeNode cur=left.pop();
        TreeNode temp=cur.left;
        while(temp!=null){
            left.push(temp);
            temp=temp.right;
        }
        return cur.val;
    }
                   
    public int getNextRight(Stack<TreeNode> right){
        TreeNode cur=right.pop();
        TreeNode temp=cur.right;
        while(temp!=null){
            right.push(temp);
            temp=temp.left;
        }
        return cur.val;
    }
	public int leafToRoot(TreeNode root) {
		if(root==null)return 0;
		if(root.left==null)return leafToRoot(root.right)+root.val;
		if(root.right==null)return leafToRoot(root.left)+root.val;
		int left=leafToRoot(root.left);
		int right=leafToRoot(root.right);
		return Math.max(left, right)+root.val;
	}
	
	public List<String> topKFrequent(String[] words, int k) {
        class Node{
            String word;
            int freq;
            Node(String word, int freq){
                this.word=word;
                this.freq=freq;
            }
        }
        Map<String, Node> map=new HashMap<>();
        for(String s: words){
            if(map.containsKey(s)){
                map.get(s).freq++;
            }else{
                map.put(s, new Node(s, 1));
            }
        }
        PriorityQueue<Node> pq=new PriorityQueue<>(new Comparator<Node>(){
            public int compare(Node n1, Node n2){
                if(n1.freq==n2.freq){
                    return n2.word.compareTo(n1.word);
                }else{
                    return n1.freq-n2.freq;
                }
            }
        });
        
        for(String s: map.keySet()){
            if(pq.size()<k){
                pq.offer(map.get(s));
            }else{
                if(pq.peek().freq<map.get(s).freq){
                    pq.poll();
                    pq.offer(map.get(s));
                }else if(pq.peek().freq==map.get(s).freq){
                    if((map.get(s).word.compareTo(pq.peek().word))<0){
                        pq.poll();
                        pq.offer(map.get(s));
                    }
                }
            }
        }
        List<String> res=new ArrayList<>();
        while(!pq.isEmpty()){
            res.add(pq.poll().word);
        }
        Collections.reverse(res);
        return res;
        // PriorityQueue<Node> pq=new PriorityQueue<>(new Comparator<Node>(){
        //     public int compare(Node n1, Node n2){
        //         if(n1.freq==n2.freq){
        //             return n1.word.compareTo(n2.word);
        //         }else{
        //             return n2.freq-n1.freq;
        //         }
        //     }
        // });
        // for(String s: map.keySet()){
        //     pq.offer(map.get(s));
        // }
        // List<String> res=new ArrayList<>();
        // for(int i=0;i<k;i++){
        //     res.add(pq.poll().word);
        // }
        // return res;
    }
	
	public double maximumAverageSubtree(TreeNode root) {
        double[] max=new double[1];
        helper(root, max);
        return max[0];
    }
    
    public int[] helper(TreeNode root, double[] max){
        if(root==null) return new int[]{0,0};
        int[] left=helper(root.left, max);
        int[] right=helper(root.right, max);
        int curSum=left[0]+right[0]+root.val;
        int num=left[1]+right[1]+1;
        double average=(double)curSum/num;
        max[0]=Math.max(max[0], average);
        return new int[]{curSum, num};
    }
    
//	public List<List<Integer>> verticalOrder2(TreeNode root) {
//		class Element {
//			TreeNode tn;
//			int index;
//
//			Element(TreeNode tn, int index) {
//				this.tn = tn;
//				this.index = index;
//			}
//		}
//		List<List<Integer>> res = new ArrayList<>();
//		if (root == null)
//			return res;
//		Queue<Element> queue = new LinkedList<>();
//		queue.offer(new Element(root, 0));
//		Map<Integer, List<Integer>> map = new HashMap<>();
//		while (!queue.isEmpty()) {
//			Element cur = queue.poll();
//			TreeNode curTn = cur.tn;
//			int curIndex = cur.index;
//			if (map.containsKey(curIndex)) {
//				map.get(curIndex).add(curTn.val);
//			} else {
//				List<Integer> list = new ArrayList<>();
//				list.add(curTn.val);
//				map.put(curIndex, list);
//			}
//			if (curTn.left != null)
//				queue.offer(new Element(curTn.left, index - 1));
//			if (curTn.right != null)
//				queue.offer(new Element(curTn.right, index + 1));
//		}
//		for (Integer i : map.keySet()) {
//			res.add(map.get(i));
//		}
//		return res;
//	}

	public static TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
		System.out.println(111);
		// return inorderSuccessorHelper(root, p, new boolean[]{false});
		return null;
	}

	public String serialize2(TreeNode root) {
		StringBuilder sb = new StringBuilder();
		helper(root, sb);
		return sb.toString();

	}

	public void helper(TreeNode root, StringBuilder sb) {
		if (root == null)
			return;
		sb.append(root.val).append(",");
		helper(root.left, sb);
		helper(root.right, sb);
	}

	// Decodes your encoded data to tree.
	public TreeNode deserialize2(String data) {
		if (data.length() == 0)
			return null;
		Queue<String> q = new LinkedList<>(Arrays.asList(data.split(",")));
		return deserializeHelper(q, Integer.MIN_VALUE, Integer.MAX_VALUE);
	}

	public TreeNode deserializeHelper(Queue<String> q, int min, int max) {
		if (q.isEmpty())
			return null;
		int cur = Integer.parseInt(q.peek());
		if (cur > max)
			return null;
		q.poll();
		TreeNode tn = new TreeNode(cur);
		tn.left = deserializeHelper(q, min, cur);
		tn.right = deserializeHelper(q, cur, max);
		return tn;
	}

	public static String numberToWords(int num) {
		if (num == 0)
			return "zero";
		String[][] dic = new String[3][];
		dic[0] = new String[] { "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine" };
		dic[1] = new String[] { "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
				"Eighteen", "Nineteen", "Twenty", "Thirty", "Fourty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" };
		dic[2] = new String[] { "", " Thousand", " Million", " Billion" };
		int count = 0;
		int temp = num;
		List<String> list = new ArrayList<>();
		while (temp > 0) {
			int cur = temp % 1000;
			list.add(compute(cur, count, dic));
			count++;
			temp /= 1000;
		}
		StringBuilder sb = new StringBuilder();
		for (int i = list.size() - 1; i >= 1; i--) {
			sb.append(list.get(i)).append(" ");
		}
		sb.append(list.get(0));
		return sb.toString();
	}

	public static String compute(int cur, int count, String[][] dic) {
		if (cur < 10) {
			return dic[0][cur] + " " + dic[2][count];
		}
		if (cur < 100) {
			int s = cur / 10;
			int g = cur % 10;
			if (s == 1) {
				return dic[1][g] + dic[2][count];
			}
			String temp = (g == 0) ? "" : (" " + dic[0][g]);
			return dic[1][s + 8] + temp + dic[2][count];
		}
		int h = cur / 100;
		int remain = cur % 100;
		return dic[0][h] + " Hundred " + compute(remain, count, dic);
	}

	public List<List<Integer>> generate2(int numRows) {
		List<List<Integer>> result = new ArrayList<>();
		// if( numRows < 0) throw new IllegalArgumentException();
		if (numRows == 0)
			return result;
		result.add(new ArrayList<>());
		result.get(0).add(1);

		for (int n = 1; n < numRows; n++) {
			List<Integer> nRow = new ArrayList<>();
			// 让nRow等于n-1Row
			nRow = result.get(n - 1);
			// 在最后index=n处加上1
			nRow.add(1);
			// 从右向左赋值index = 【n-1，1】（0位不用改 直接是1，n位为1）
			int N = n - 1;
			while (N > 0) {
				nRow.set(N, nRow.get(N - 1) + nRow.get(N));
				N--;
			}
			result.add(nRow);
		}
		return result;
	}

	public int longestConsecutive2(TreeNode root) {
		return longestConsecutive2Helper(root)[2];
	}

	public int[] longestConsecutive2Helper(TreeNode root) {
		if (root == null) {
			return new int[] { 0, 0, 0 };// one bigger, one smaller, one max
		}
		int[] left = longestConsecutive2Helper(root.left);
		int[] right = longestConsecutive2Helper(root.right);
		int[] res = new int[] { 1, 1, Math.max(left[2], right[2]) };
		if (root.left != null) {
			if (root.left.val == root.val + 1) {
				res[0] = left[0] + 1;
			} else if (root.left.val == root.val - 1) {
				res[1] = left[1] + 1;
			}
		}
		if (root.right != null) {
			if (root.right.val == root.val + 1) {
				res[0] = Math.max(res[0], right[0] + 1);
			} else if (root.right.val == root.val - 1) {
				res[1] = Math.max(res[1], right[1] + 1);
			}
		}
		res[2] = Math.max(res[2], res[0] + res[1] - 1);
		return res;
	}

	public int longestConsecutive(TreeNode root) {
		return longestConsecutiveHelper(root)[1];
	}

	public int[] longestConsecutiveHelper(TreeNode root) {
		if (root == null) {
			return new int[] { 0, 0 };
		}
		int[] left = longestConsecutiveHelper(root.left);
		int[] right = longestConsecutiveHelper(root.right);
		int curL = 1, curR = 1;
		if (root.left != null) {
			if (root.val == root.left.val - 1) {
				curL = left[0] + 1;
			}
		}
		if (root.right != null) {
			if (root.val == root.right.val - 1) {
				curR = right[0] + 1;
			}
		}
		int max = Math.max(curL, curR);
		int[] res = new int[] { max, Math.max(max, Math.max(left[1], right[1])) };
		return res;
	}

	public int countUnivalSubtrees2(TreeNode root) {
		int[] count = { 0 };
		helper(root, count);
		return count[0];
	}

	public boolean helper(TreeNode root, int[] count) {
		if (root == null)
			return true;
		boolean left = helper(root.left, count);
		boolean right = helper(root.right, count);
		if (left && right) {
			if (root.left != null && root.left.val != root.val || root.right != null && root.right.val != root.val) {
				return false;
			} else {
				count[0]++;
				return true;
			}
		} else {
			return false;
		}
	}

	public int countNodes(TreeNode root) {
		if (root == null)
			return 0;
		int heightL = getHeight(root.left);
		int heightR = getHeight(root.right);
		int count = 1;
		if (heightL > heightR) {// right tree is a full tree
			count += (1 << heightR) - 1 + countNodes(root.left);
		} else if (heightL == heightR) {// left tree is a full tree
			count += (1 << heightL) - 1 + countNodes(root.right);
		} else {
			return -1;
		}
		return count;
	}

	public int getHeight(TreeNode root) {
		int height = 0;
		while (root != null) {
			root = root.left;
			height++;
		}
		return height;
	}

	public List<TreeNode> generateTrees2(int n) {
		if (n <= 0)
			return new ArrayList<>();
		Map<String, List<TreeNode>> map = new HashMap<>();
		return generateTrees2helper(1, n, map);
	}

	public List<TreeNode> generateTrees2helper(int start, int end, Map<String, List<TreeNode>> map) {
		List<TreeNode> res = new ArrayList<>();
		if (start > end) {
			res.add(null);
			return res;
		}
		String s = start + "," + end;
		if (map.containsKey(s))
			return map.get(s);
		for (int i = start; i <= end; i++) {
			List<TreeNode> left = generateTrees2helper(start, i - 1, map);
			List<TreeNode> right = generateTrees2helper(i + 1, end, map);
			for (TreeNode l : left) {
				for (TreeNode r : right) {
					TreeNode root = new TreeNode(i);
					root.left = l;
					root.right = r;
					res.add(root);
				}
			}
		}
		map.put(s, res);
		return res;
	}

	public String GroupAnagram(String str) {
		Map<Character, Integer> map = new HashMap<>();
		char[] ch = str.toCharArray();
		int sum = 0;
		for (char c : ch) {
			sum += pow(c - 'a');
		}
		return null;
	}

	public int pow(int n) {
		return 1;
	}

	public static boolean canFinish(int numCourses, int[][] prerequisites) {
		List<Integer>[] graph = new ArrayList[numCourses];
		for (int i = 0; i < numCourses; i++) {
			graph[i] = new ArrayList<>();
		}
		for (int[] cur : prerequisites) {
			graph[cur[1]].add(cur[0]);
		}
		int[] status = new int[numCourses];
		for (int i = 0; i < numCourses; i++) {
			if (!isValid(i, graph, status)) {
				return false;
			}
		}
		return true;
	}

	public static boolean isValid(int source, List<Integer>[] graph, int[] status) {
		if (status[source] == 2)
			return true;
		if (status[source] == 1)
			return false;

		status[source] = 1;
		List<Integer> list = graph[source];
		for (Integer i : list) {
			if (!isValid(i, graph, status)) {
				return false;
			}
		}
		status[source] = 2;
		return true;
	}

	public String mostCommonWord(String paragraph, String[] banned) {
		char[] ch = paragraph.toCharArray();
		int len = ch.length;
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < len; i++) {
			if (ch[i] >= 'a' && ch[i] <= 'z' || ch[i] == ' ') {
				sb.append(ch[i]);
			} else if (ch[i] >= 'A' && ch[i] <= 'Z') {
				sb.append((char) (ch[i] + 32));
			} else {
				sb.append(" ");
			}
		}
		String[] s = sb.toString().split(" ");
		Set<String> set = new HashSet<>();
		for (String ss : banned) {
			set.add(ss);
		}
		Map<String, Integer> map = new HashMap<>();
		for (String sss : s) {
			if (!set.contains(sss) && !sss.equals("")) {
				map.put(sss, map.getOrDefault(sss, 0) + 1);
			}
		}
		int max = 0;
		String res = "";
		for (String str : map.keySet()) {
			if (map.get(str) > max) {
				max = map.get(str);
				res = str;
			}
		}
		return res;
	}

	public int[][] kClosest(int[][] points, int K) {
		PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				// TODO Auto-generated method stub
				return o2[1] * o2[1] + o2[0] * o2[0] - o1[1] * o1[1] - o1[0] * o1[0];
			}
		});
		for (int i = 0; i < K; i++) {
			pq.offer(points[i]);
		}
		int len = points.length;
		for (int i = K; i < len; i++) {
			pq.offer(points[i]);
			pq.poll();
		}
		int[][] res = new int[K][2];
		int i = 0;
		while (!pq.isEmpty()) {
			int[] cur = pq.poll();
			res[i][0] = cur[0];
			res[i][1] = cur[1];
			i++;
		}
		return res;
	}

	public static List<String> numberDistinct(String s, int k) {
		char[] ch = s.toCharArray();
		List<Integer> res = new ArrayList<>();
		Set<String> set = new HashSet<>();
		int[] count = new int[26];
		int num = 0;
		int i = 0;
		for (; i < k; i++) {
			count[ch[i] - 'a']++;
			if (count[ch[i] - 'a'] == 1) {
				num++;
			}
		}
		if (num == k)
			set.add(s.substring(0, k));
		int len = ch.length;
		for (; i < len; i++) {
			if (++count[ch[i] - 'a'] == 1)
				num++;
			if (num > k) {
				if (--count[ch[i - k + 1] - 'a'] == 0)
					num--;
			}
			if (num == k)
				set.add(s.substring(i - k + 1, i + 1));
		}
		return new ArrayList<String>(set);
	}

	public static String longestPalindrome3(String s) {
		char[] ch = s.toCharArray();
		int len = ch.length;
		boolean[][] dp = new boolean[len][len];
		int left = -1, right = -1;
		for (int l = len - 1; l >= 0; l--) {
			for (int r = l; r < len; r++) {
				if (ch[l] == ch[r] && (r - l <= 2 || dp[l + 1][r - 1])) {
					dp[l][r] = true;
					if (left == -1 || (r - l > right - left)) {
						right = r;
						left = l;
					}
				}
			}
		}
		if (left == -1)
			return "";
		return s.substring(left, right + 1);
	}

	public static int maxPathScore(int[][] array) {
		int row = array.length, col = array[0].length;
		int[][] dp = new int[row][col];
		dp[0][0] = Integer.MAX_VALUE;
		dp[row - 1][col - 1] = Integer.MAX_VALUE;
		for (int i = 1; i < row; i++) {
			dp[i][0] = Math.min(dp[i - 1][0], array[i][0]);
		}
		for (int j = 1; j < col; j++) {
			dp[0][j] = Math.min(dp[0][j - 1], array[0][j]);
		}
		for (int i = 1; i < row; i++) {
			for (int j = 1; j < col; j++) {
				if (i == row - 1 && j == col - 1)
					return Math.max(dp[i - 1][j], dp[i][j - 1]);
				int a = Math.min(array[i][j], dp[i - 1][j]);
				int b = Math.min(array[i][j], dp[i][j - 1]);
				dp[i][j] = Math.max(a, b);
			}
		}
		return dp[row - 1][col - 1];
	}

	public static int subarraysWithKDistinct2(int[] A, int K) {
		if (A == null || A.length == 0)
			return 0;
		return atMost2(A, K) - atMost2(A, K - 1);
	}

	public static int atMost2(int[] A, int K) {
		Map<Integer, Integer> map = new HashMap<>();
		int len = A.length;
		int count = 0, res = 0;
		for (int i = 0, j = 0; j < len; j++) {
			map.put(A[j], map.getOrDefault(A[j], 0) + 1);
			if (map.get(A[j]) == 1)
				count++;
			while (count > K) {
				map.put(A[i], map.get(A[i]) - 1);
				if (map.get(A[i]) == 0) {
					count--;
				}
				i++;
			}
			res += j - i + 1;
		}
		return res;
	}

	class UnionFind {
		int size, row, col;
		int[] ids, sz;

		UnionFind(int row, int col) {
			size = 0;
			this.row = row;
			this.col = col;
			int len = row * col;
			ids = new int[len];
			sz = new int[len];
			for (int i = 0; i < len; i++) {
				ids[i] = -1;
				sz[i] = 1;
			}
		}

		boolean find(int p, int q) {
			return getRoot(p) == getRoot(q);
		}

		void union(int p, int q) {
			int pRoot = getRoot(p), qRoot = getRoot(q);
			if (sz[pRoot] > sz[qRoot]) {
				ids[qRoot] = pRoot;
				sz[pRoot] += sz[qRoot];
			} else {
				ids[pRoot] = qRoot;
				sz[qRoot] += sz[pRoot];
			}
			size--;
		}

		int getRoot(int p) {
			while (ids[p] != p) {
				ids[p] = ids[ids[p]];
				p = ids[p];
			}
			return p;
		}

		void addIsland(int i, int j) {
			int index = i * col + j;
			if (ids[index] == -1) {
				ids[index] = index;
				size++;
			}
		}
	}

	public List<Integer> numIslands222(int m, int n, int[][] positions) {
		List<Integer> res = new ArrayList<>();
		UnionFind uf = new UnionFind(m, n);
		int[][] directions = new int[][] { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };
		for (int[] pos : positions) {
			uf.addIsland(pos[0], pos[1]);
			int temp = pos[0] * n + pos[1];
			for (int[] dir : directions) {
				int i = pos[0] + dir[0];
				int j = pos[1] + dir[1];
				int index = i * n + j;
				if (i >= 0 && i < m && j >= 0 && j < n && uf.ids[index] != -1 && !uf.find(temp, index)) {
					uf.union(temp, index);
				}
			}
			res.add(uf.size);
		}
		return res;
	}

	public List<Integer> numIslands2(int m, int n, int[][] positions) {
		List<Integer> res = new LinkedList<>();
		if (m <= 0 || n <= 0) {
			return res;
		}
		int count = 0;
		int[] roots = new int[m * n]; // 1D array of roots
		int[] size = new int[m * n]; // 1D array of size of each tree
		Arrays.fill(roots, -1); // Every position is water initially.
		int[][] directions = new int[][] { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
		for (int[] p : positions) {
			int island = p[0] * n + p[1];
			if (roots[island] == -1) {
				roots[island] = island; // Set it to be the root of itself.
				size[island]++;
				count++;
			}
			// Check four directions
			for (int[] dir : directions) {
				int x = p[0] + dir[0], y = p[1] + dir[1];
				int neighbor = x * n + y;
				// Skip when x or y is invalid, or neighbor is water.
				if (x < 0 || x >= m || y < 0 || y >= n || roots[neighbor] == -1) {
					continue;
				}
				int neighborRoot = find(neighbor, roots);
				int islandRoot = find(island, roots);
				if (islandRoot != neighborRoot) {
					// Union by size
					if (size[islandRoot] >= size[neighborRoot]) {
						size[islandRoot] += size[neighborRoot];
						roots[neighborRoot] = islandRoot;
					} else {
						size[neighborRoot] += size[islandRoot];
						roots[islandRoot] = neighborRoot;
					}
					count--;
				}
			}
			res.add(count);
		}
		return res;
	}

	private int find(int id, int[] roots) {
		while (roots[id] != id) {
			roots[id] = roots[roots[id]];
			// roots[id] = find(roots[id], roots); // path compression
			id = roots[id];
		}
		return id;
	}

	public List<Integer> rightSideView(TreeNode root) {
		List<Integer> res = new ArrayList<>();
		if (root == null)
			return res;
		Queue<TreeNode> queue = new LinkedList<>();
		queue.offer(root);
		while (!queue.isEmpty()) {
			int size = queue.size();
			TreeNode cur = null;
			while (size-- > 0) {
				cur = queue.poll();
				if (cur.left != null) {
					queue.offer(cur.left);
				}
				if (cur.right != null) {
					queue.offer(cur.right);
				}
			}
			res.add(cur.val);
		}
		return res;
	}

	public static int calculate4(String s) {
		char[] ch = s.toCharArray();
		int res = 0, sum = 0;
		char sign = '+';
		Stack<Integer> stack = new Stack<>();
		int len = ch.length;
		for (int i = 0; i < len; i++) {
			if (ch[i] == '(') {
				int l = 1;
				int j = i + 1;
				while (j < len) {
					if (ch[j] == '(') {
						l++;
					} else if (ch[j] == ')') {
						if (--l == 0)
							break;
					}
					j++;
				}
				String temp = s.substring(i + 1, j);
				int blockVal = calculate4(temp);
				i = j;
				helper(stack, sign, blockVal);
			} else if (Character.isDigit(ch[i])) {
				sum = sum * 10 + ch[i] - '0';
				while (i < len - 1 && Character.isDigit(ch[i + 1])) {
					sum = sum * 10 + ch[i + 1] - '0';
					i++;
				}
				helper(stack, sign, sum);
				sum = 0;
			} else if (ch[i] == '+') {
				sign = '+';
			} else if (ch[i] == '-') {
				sign = '-';
			} else if (ch[i] == '*') {
				sign = '*';
			} else if (ch[i] == '/') {
				sign = '/';
			}
		}
		while (!stack.isEmpty()) {
			res += stack.pop();
		}
		return res;
	}

	public static void helper(Stack<Integer> stack, char sign, int sum) {
		if (sign == '+') {
			stack.push(sum);
		} else if (sign == '-') {
			stack.push(-sum);
		} else if (sign == '*') {
			stack.push(stack.pop() * sum);
		} else if (sign == '/') {
			stack.push(stack.pop() / sum);
		}
	}

	public int calculate3(String s) {
		char[] ch = s.toCharArray();
		int res = 0, sum = 0;
		char sign = '+';
		Stack<Integer> stack = new Stack<>();
		int len = ch.length;
		for (int i = 0; i < len; i++) {
			if (Character.isDigit(ch[i])) {
				sum = sum * 10 + ch[i] - '0';
				while (i < len - 1 && Character.isDigit(ch[i + 1])) {
					sum = sum * 10 + ch[i + 1] - '0';
					i++;
				}
				if (sign == '+') {
					stack.push(sum);
				} else if (sign == '-') {
					stack.push(-sum);
				} else if (sign == '*') {
					stack.push(stack.pop() * sum);
				} else if (sign == '/') {
					stack.push(stack.pop() / sum);
				}
				sum = 0;
			} else if (ch[i] == '+') {
				sign = '+';
			} else if (ch[i] == '-') {
				sign = '-';
			} else if (ch[i] == '*') {
				sign = '*';
			} else if (ch[i] == '/') {
				sign = '/';
			}
		}
		while (!stack.isEmpty()) {
			res += stack.pop();
		}
		return res;
	}

	public static int calculate2(String s) {
		char[] ch = s.toCharArray();
		int res = 0, sum = 0, sign = 1;
		Stack<Integer> stack = new Stack<>();
		int len = ch.length;
		for (int i = 0; i < len; i++) {
			if (Character.isDigit(ch[i])) {
				sum = sum * 10 + ch[i] - '0';
				while (i < len - 1 && Character.isDigit(ch[i + 1])) {
					sum = sum * 10 + ch[i + 1] - '0';
					i++;
				}
				res += sign * sum;
				sum = 0;
			} else if (ch[i] == '+') {
				sign = 1;
			} else if (ch[i] == '-') {
				sign = -1;
			} else if (ch[i] == '(') {
				stack.push(res);
				stack.push(sign);
				res = 0;
				sign = 1;
			} else if (ch[i] == ')') {
				res = res * stack.pop() + stack.pop();
			}
		}
		return res;
	}

	public String fractionToDecimal(int numerator, int denominator) {
		if (numerator == 0)
			return "0";
		StringBuilder sb = new StringBuilder();
		if (numerator > 0 && denominator < 0 || (numerator < 0 && denominator > 0)) {
			sb.append("-");
		}
		long num = Math.abs((long) numerator);
		long den = Math.abs((long) denominator);
		long res = num / den;
		sb.append(res);
		num %= den;
		if (num == 0)
			return sb.toString();
		sb.append(".");
		Map<Long, Integer> map = new HashMap<>();
		while (num != 0) {
			num *= 10;
			res = num / den;
			sb.append(res);
			num %= den;
			if (map.containsKey(num)) {
				sb.insert(map.get(num), "(");
				sb.append(")");
				return sb.toString();
			} else {
				map.put(num, sb.length());
			}
		}
		return sb.toString();
	}

	public int findDuplicate(int[] nums) {
		int len = nums.length;
		int[] count = new int[len - 1];
		for (int i : nums) {
			if (++count[i - 1] == 2) {
				return i;
			}
		}
		return 0;
	}

	public int findMin(int[] nums) {
		int left = 0, right = nums.length - 1;
		while (left < right - 1) {
			int mid = left + (right - left) / 2;
			if (nums[mid] > nums[0]) {
				left = mid;
			} else {
				right = mid;
			}
		}
		if (nums[right] > nums[0])
			return nums[0];
		return nums[right];
	}

	public int findKthNumber(int m, int n, int k) {
		PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
			public int compare(int[] o1, int[] o2) {
				return o1[2] - o2[2];
			}
		});
		for (int j = 0; j < n && j < k; j++) {
			pq.offer(new int[] { 0, j, j + 1 });
		}
		int count = 0;
		while (!pq.isEmpty()) {
			int[] cur = pq.poll();
			if (++count == k)
				return cur[2];
			if (cur[0] == m - 1)
				continue;
			if (cur[1] == 0) {
				pq.offer(new int[] { cur[0] + 1, cur[1], cur[0] + 1 });
			} else {
				pq.offer(new int[] { cur[0] + 1, cur[1], (cur[0] + 2) * (cur[1] + 1) });
			}
		}
		return 0;
	}

	public static int kthSmallest(int[][] matrix, int k) {
		class Tuple {
			int x;
			int y;
			int val;

			Tuple(int x, int y, int val) {
				this.x = x;
				this.y = y;
				this.val = val;
			}
		}
		int row = matrix.length;
		int col = matrix[0].length;
		PriorityQueue<Tuple> pq = new PriorityQueue<>(new Comparator<Tuple>() {
			@Override
			public int compare(Tuple o1, Tuple o2) {
				return o1.val - o2.val;
			}
		});
		for (int i = 0; i < col && i < k; i++) {
			pq.offer(new Tuple(0, i, matrix[0][i]));
		}
		int count = 0;
		while (!pq.isEmpty()) {
			Tuple cur = pq.poll();
			if (++count == k)
				return matrix[cur.x][cur.y];
			if (cur.x == row - 1)
				continue;
			pq.offer(new Tuple(cur.x + 1, cur.y, matrix[cur.x + 1][cur.y]));
		}
		return 0;
	}

	public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
		List<List<Integer>> res = new ArrayList<>();
		if (nums1 == null || nums1.length == 0 || nums2 == null || nums2.length == 0)
			return res;
		PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				return o1[0] + o1[1] - o2[0] - o2[1];
			}
		});
		for (int i = 0; i < nums1.length && i < k; i++) {
			pq.offer(new int[] { nums1[i], nums2[0], 0 });
		}
		int count = 0;
		while (!pq.isEmpty()) {
			int[] cur = pq.poll();
			res.add(Arrays.asList(cur[0], cur[1]));
			if (++count == k)
				break;
			if (cur[2] == nums2.length - 1)
				continue;
			pq.offer(new int[] { cur[0], nums2[cur[2] + 1], cur[2] + 1 });
		}
		return res;
	}

	public static List<String> generatePalindromes(String s) {
		List<String> res = new ArrayList<>();
		int[] count = new int[256];
		char[] ch = s.toCharArray();
		for (int i = 0; i < ch.length; i++) {
			count[ch[i]]++;
		}
		int k = 0;
		String mid = "";
		List<Character> list = new ArrayList<>();
		for (int i = 0; i < 256; i++) {
			if (count[i] > 0) {
				if (count[i] % 2 != 0) {
					mid = "" + (char) i;
					if (++k > 1)
						return res;
				}
				for (int j = count[i] / 2; j > 0; j--) {
					list.add((char) i);
				}
			}
		}
		generatePalindromesHelper(list, mid, res, new StringBuilder(), new boolean[list.size()]);
		return res;
	}

	public static void generatePalindromesHelper(List<Character> list, String mid, List<String> res, StringBuilder sb,
			boolean[] used) {
		if (sb.length() == list.size()) {
			res.add(sb.toString() + mid + sb.reverse().toString());
			sb.reverse();
			return;
		}
		int size = list.size();
		for (int i = 0; i < size; i++) {
			if (used[i] || i > 0 && list.get(i) == list.get(i - 1) && !used[i - 1])
				continue;
			used[i] = true;
			sb.append(list.get(i));
			generatePalindromesHelper(list, mid, res, sb, used);
			sb.deleteCharAt(sb.length() - 1);
			used[i] = false;
		}
	}

	public static boolean canPermutePalindrome2(String s) {
		char[] ch = s.toCharArray();
		int len = ch.length;
		Map<Character, Integer> map = new HashMap<>();
		for (int i = 0; i < len; i++) {
			int temp = map.getOrDefault(ch[i], 0) + 1;
			map.put(ch[i], temp);
		}
		int k = 0;
		for (Character c : map.keySet()) {
			if (map.get(c) % 2 != 0) {
				if (++k > 1)
					return false;
			}
		}
		return true;
	}

	public static int minTransfers(int[][] transactions) {
		Map<Integer, Integer> map = new HashMap<>();
		for (int[] one : transactions) {
			map.put(one[0], map.getOrDefault(one[0], 0) + one[2]);
			map.put(one[1], map.getOrDefault(one[1], 0) - one[2]);
		}
		List<Integer> list = new ArrayList<>(map.values());
		int[] num = { Integer.MAX_VALUE };
//        dfs(list, 0, 0, num);
//        return num[0];
		return dfs(list, 0);
	}

	public static int dfs(List<Integer> list, int index) {
		if (index == list.size())
			return 0;
		int begin;
		for (begin = index; begin < list.size(); begin++) {
			if (list.get(begin) != 0)
				break;
		}
		if (begin == list.size())
			return 0;
		int temp = list.get(begin);
		int val = Integer.MAX_VALUE;
		for (int j = begin + 1; j < list.size(); j++) {
			if (temp * list.get(j) < 0) {
				list.set(j, list.get(j) + temp);
//				list.set(begin, 0);
				val = Math.min(val, 1 + dfs(list, begin + 1));
//				list.set(begin, temp);
				list.set(j, list.get(j) - temp);
			}
		}
		return val;
	}

	public static void dfs(List<Integer> list, int index, int cur, int[] num) {
		int begin = index;
		for (begin = index; begin < list.size(); begin++) {
			if (list.get(begin) != 0) {
				break;
			}
		}
		if (begin == list.size()) {
			num[0] = Math.min(num[0], cur);
			return;
		}
		int temp = list.get(begin);
		for (int j = begin + 1; j < list.size(); j++) {
			if (temp * list.get(j) < 0) {
				list.set(j, list.get(j) + temp);
				list.set(begin, 0);
				dfs(list, index + 1, cur + 1, num);
				list.set(begin, temp);
				list.set(j, list.get(j) - temp);
			}
		}
	}

	public static int calculate(String s) {
		char[] ch = s.toCharArray();
		int[] res = { 0 };
		calculateHelper(ch, 0, 0, 0, '+', res);
		return res[0];
	}

	public static void calculateHelper(char[] ch, int index, int val, int prev, char sign, int[] res) {
		if (index == ch.length) {
			if (sign == '+') {
				res[0] += val;
			}
			if (sign == '-') {
				res[0] -= val;
			}
			if (sign == '*') {
				res[0] += -prev + prev * val;
			}
			if (sign == '/') {
				res[0] += -prev + prev / val;
			}
		} else if (ch[index] == ' ') {
			calculateHelper(ch, index + 1, val, prev, sign, res);
		} else if (Character.isDigit(ch[index])) {
			calculateHelper(ch, index + 1, val * 10 + ch[index] - '0', prev, sign, res);
		} else {
			if (sign == '+') {
				res[0] += val;
				calculateHelper(ch, index + 1, 0, val, ch[index], res);
			}
			if (sign == '-') {
				res[0] -= val;
				calculateHelper(ch, index + 1, 0, -val, ch[index], res);
			}
			if (sign == '*') {
				res[0] += -prev + prev * val;
				calculateHelper(ch, index + 1, 0, prev * val, ch[index], res);
			}
			if (sign == '/') {
				res[0] += -prev + prev / val;
				calculateHelper(ch, index + 1, 0, prev / val, ch[index], res);
			}
		}
	}

	public static List<String> findItinerary(List<List<String>> tickets) {
		Map<String, List<String>> graph = new HashMap<>();
		for (List<String> list : tickets) {
			if (graph.containsKey(list.get(0))) {
				graph.get(list.get(0)).add(list.get(1));
			} else {
				List<String> temp = new ArrayList<>();
				temp.add(list.get(1));
				graph.put(list.get(0), temp);
			}
		}
		for (String s : graph.keySet()) {
			Collections.sort(graph.get(s));
		}
		List<String> res = new ArrayList<>();
		res.add("JFK");
		int num = tickets.size() + 1;
		findItineraryHelper(res, graph, "JFK", num);
		return res;
	}

	public static boolean findItineraryHelper(List<String> res, Map<String, List<String>> graph, String str, int num) {
		if (res.size() == num)
			return true;
		List<String> list = graph.get(str);
		if (list == null || list.size() == 0)
			return false;
		for (int i = 0; i < list.size(); i++) {
			String s = list.remove(i);
			res.add(s);
			if (findItineraryHelper(res, graph, s, num))
				return true;
			list.add(i, s);
			res.remove(res.size() - 1);
		}
		return false;
	}

	public static int[][] intervalIntersection(int[][] A, int[][] B) {
		class Node {
			int start;
			int end;

			Node(int start, int end) {
				this.start = start;
				this.end = end;
			}
		}
		List<Node> list = new ArrayList<>();
		int pA = 0, pB = 0;
		int vA, vB;
		int lenA = A.length, lenB = B.length;
		while (pA < lenA && pB < lenB) {
			if (A[pA][1] < B[pB][1]) {
				if (B[pB][0] <= A[pA][0]) {
					list.add(new Node(A[pB][0], A[pA][1]));
				} else if (B[pB][0] <= A[pA][1]) {
					list.add(new Node(B[pB][0], A[pA][1]));
				}
				pA++;
			} else if (A[pA][1] > B[pB][1]) {
				if (A[pA][0] <= B[pB][0]) {
					list.add(new Node(B[pB][0], B[pB][1]));
				} else if (A[pA][0] <= B[pB][1]) {
					list.add(new Node(A[pA][0], B[pB][1]));

					pB++;
				} else {
					if (A[pA][0] > B[pB][0]) {
						list.add(new Node(A[pA][0], A[pA][1]));
					} else {
						list.add(new Node(B[pB][0], A[pA][1]));
					}
					pA++;
					pB++;
				}
			}
		}
		int[][] res = new int[list.size()][2];
		int index = 0;
		for (Node n : list) {
			res[index][0] = n.start;
			res[index++][1] = n.end;
		}
		return res;
	}

	public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {
		Map<TreeNode, TreeNode> map = new HashMap<>();
		dfs(root, null, map);
		Queue<TreeNode> queue = new LinkedList<>();
		Set<TreeNode> set = new HashSet<>();
		List<Integer> list = new ArrayList<>();
		set.add(target);
		queue.add(target);
		int level = 0;
		while (!queue.isEmpty()) {
			int size = queue.size();
			if (level == K) {
				break;
			}
			while (size-- > 0) {
				TreeNode cur = queue.poll();
				if (cur.left != null && set.add(cur.left)) {
					queue.offer(cur.left);
				}
				if (cur.right != null && set.add(cur.right)) {
					queue.offer(cur.right);
				}
				if (map.get(cur) != null && set.add(map.get(cur))) {
					queue.offer(map.get(cur));
				}
			}
			level++;
		}
		for (TreeNode treenode : queue) {
			list.add(treenode.val);
		}
		return list;
	}

	public void dfs(TreeNode cur, TreeNode par, Map<TreeNode, TreeNode> map) {
		if (cur == null)
			return;
		map.put(cur, par);
		dfs(cur.left, cur, map);
		dfs(cur.right, cur, map);
	}

	public boolean canMeasureWater(int x, int y, int z) {
		int total = x + y;
		if (z < 0 || z > total)
			return false;
		Set<Integer> set = new HashSet<>();
		Queue<Integer> queue = new LinkedList<>();
		queue.offer(0);
		while (!queue.isEmpty()) {
			int cur = queue.poll();
			int temp = cur + x;
			if (temp <= total && set.add(temp)) {
				queue.offer(temp);
			}
			temp = cur + y;
			if (temp <= total && set.add(temp)) {
				queue.offer(temp);
			}
			temp = cur - x;
			if (temp >= 0 && set.add(temp)) {
				queue.offer(temp);
			}
			temp = cur - y;
			if (temp >= 0 && set.add(temp)) {
				queue.offer(temp);
			}
			if (set.contains(z))
				return true;
		}
		return false;
	}

	public static double new21Game3(int N, int K, int W) {
		if (K == 0 || N >= K + W)
			return 1.0;
		double[] dp = new double[N + 1];
		dp[0] = 1.0;
		double sum = 1, res = 0;
		for (int i = 1; i <= N; i++) {
			dp[i] = sum / W;
			if (i < K) {
				sum += dp[i];
			} else {
				res += dp[i];
			}
			if (i >= W)
				sum -= dp[i - W];
		}
		return res;
	}

//	public static double new21Game(int N, int K, int W) {
//		return 1 - new21GameHelper(N, K, W, 0, 0);
//	}

//	public static double new21GameHelper(int N, int K, int W, int level, int sum) {
//		if (sum >= K) {
//			if (sum <= N) {
//				return pow3((double) 1 / W, level);
//			} else {
//				return 0;
//			}
//		}
//		for (int i = 1; i <= W; i++) {
//			return new21GameHelper(N, K, W, level + 1, sum + i);
//		}
//	}

	private static double pow3(double x, int n) {
		if (n == 1)
			return x;
		double y = pow3(x, n / 2);
		return n % 2 == 0 ? y * y : y * y * x;
	}

	public int[] sortedSquares(int[] A) {
		int right = 0;
		int len = A.length;
		for (int i = 0; i < len; i++) {
			if (A[i] >= 0) {
				right = i;
				break;
			}
		}
		int left = right - 1;
		int[] res = new int[len];
		int index = 0;
		while (left >= 0 && right < len) {
			if (A[right] > -A[left]) {
				res[index++] = A[left] * A[left];
				left--;
			} else {
				res[index++] = A[right] * A[right];
				right++;
			}
		}
		while (left >= 0) {
			res[index++] = A[left] * A[left];
			left--;
		}
		while (right < len) {
			res[index++] = A[right] * A[right];
			right++;
		}
		return res;
	}

	public void moveZeroes(int[] nums) {
		int slow = 0;
		int len = nums.length;
		for (int i = 0; i < len; i++) {
			if (nums[i] == 0) {
				slow = i;
				break;
			}
		}
		for (int fast = slow; fast < len; fast++) {
			if (nums[fast] != 0) {
				swap2(nums, slow, fast);
				slow++;
			}
		}
	}

	public void swap2(int[] nums, int i, int j) {
		int temp = nums[i];
		nums[i] = nums[j];
		nums[j] = temp;
	}

	public int maxProfit2(int[] prices) {
		int res = 0;
		int len = prices.length;
		for (int i = 1; i < len; i++) {
			int temp = prices[i] - prices[i - 1];
			if (temp > 0) {
				res += temp;
			}
		}
		return res;
	}

	public int maxProfit(int[] prices) {
		int maxCur = 0, maxRes = 0;
		int len = prices.length;
		for (int i = 1; i < len; i++) {
			maxCur += prices[i] - prices[i - 1];
			if (maxCur < 0)
				maxCur = 0;
			maxRes = Math.max(maxRes, maxCur);
		}
		return maxRes;
	}

	public List<List<String>> groupAnagrams2(String[] strs) {
		Map<String, List<String>> map = new HashMap<>();
		List<String> list = new ArrayList<>();
		for (String s : strs) {
			char[] ch = s.toCharArray();
			Arrays.sort(ch);
			String temp = new String(ch);
			list = map.getOrDefault(temp, new ArrayList<>());
			list.add(s);
			map.put(temp, list);
		}
		List<List<String>> res = new ArrayList<>();
		for (Map.Entry<String, List<String>> entry : map.entrySet()) {
			res.add(entry.getValue());
		}
		return res;
	}

	public void solveSudoku(char[][] board) {
		boolean[][] row = new boolean[9][10];
		boolean[][] col = new boolean[9][10];
		boolean[][] subSquare = new boolean[9][10];
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (board[i][j] != '.') {
					row[i][board[i][j] - '0'] = true;
					col[j][board[i][j] - '0'] = true;
					int num = i / 3 * 3 + j / 3;
					subSquare[num][board[i][j] - '0'] = true;
				}
			}
		}
		solveSudokuHelper(board, 0, 0, row, col, subSquare);
	}

	public boolean solveSudokuHelper(char[][] board, int i, int j, boolean[][] row, boolean[][] col,
			boolean[][] subSquare) {
		if (i == 9 && j == 0)
			return true;
		if (board[i][j] == '.') {
			int num = i / 3 * 3 + j / 3;
			for (int index = 1; index <= 9; index++) {
				if (!row[i][index] && !col[j][index] && !subSquare[num][index]) {
					row[i][index] = true;
					col[j][index] = true;
					subSquare[num][index] = true;
					board[i][j] = (char) (index + '0');
					if (solveSudokuHelper(board, j == 8 ? i + 1 : i, j == 8 ? 0 : j + 1, row, col, subSquare))
						return true;
					row[i][index] = false;
					col[j][index] = false;
					subSquare[num][index] = false;
					board[i][j] = '.';
				}
			}
			return false;
		}
		return solveSudokuHelper(board, j == 8 ? i + 1 : i, j == 8 ? 0 : j + 1, row, col, subSquare);
	}

	public int characterReplacement(String s, int k) {
		int[] count = new int[26];
		char[] ch = s.toCharArray();
		int len = ch.length;
		int maxNum = 0;
		int slow = 0;
		int maxLength = 0;
		for (int i = 0; i < len; i++) {
			maxNum = Math.max(maxNum, ++count[ch[i] - 'A']);
			if (i - slow + 1 - maxNum > k) {
				count[ch[slow] - 'A']--;
				slow++;
			}
			int temp = i - slow + 1;
			if (temp > maxLength)
				maxLength = temp;
		}
		return maxLength;
	}

	public int longestOnes(int[] A, int K) {
		int slow = 0, fast = 0;
		int count = 0, max = 0;
		int len = A.length;
		while (fast < len) {
			if (A[fast] == 0) {
				count++;
			}
			fast++;
			while (count > K) {
				if (A[slow++] == 0) {
					count--;
				}
			}
			int dis = fast - slow;
			if (dis > max) {
				max = dis;
			}
		}
		return max;
	}

	public int findMaxConsecutiveOnes2(int[] nums) {
		int slow = 0, fast = 0;
		int count = 0, max = 0;
		int len = nums.length;
		while (fast < len) {
			if (nums[fast] == 0) {
				count++;
			}
			fast++;
			while (count > 1) {
				if (nums[slow++] == 0) {
					count--;
				}
			}
			int dis = fast - slow;
			if (dis > max) {
				max = dis;
			}
		}
		return max;
	}

	public int findMaxConsecutiveOnes(int[] nums) {
		int max = 0, cur = 0;
		int len = nums.length;
		for (int i = 0; i < len; i++) {
			if (nums[i] == 1) {
				cur++;
				if (max < cur) {
					max = cur;
				}
			} else {
				cur = 0;
			}
		}
		return max;
	}

	public List<Integer> transformArray(int[] arr) {
		boolean flag = false;
		List<Integer> res = new ArrayList<>();
		int len = arr.length;
		while (true) {
			int[] ans = arr.clone();
			for (int i = 1; i < len - 1; i++) {
				if (ans[i] < ans[i - 1] && ans[i] < ans[i + 1]) {
					arr[i]++;
					flag = true;
				} else if (ans[i] > ans[i - 1] && ans[i] > ans[i + 1]) {
					arr[i]--;
					flag = true;
				}
			}
			if (!flag) {
				for (int i : arr) {
					res.add(i);
				}
				return res;
			}
			flag = false;
		}
	}

	public int addDigits(int num) {
		if (num < 10)
			return num;
		int sum = 0;
		while (num > 0) {
			int rightDigit = num % 10;
			num /= 10;
			sum += rightDigit;
		}
		return addDigits(sum);
	}

	public boolean isHappy2(int n) {
		if (n == 1 || n == 7)
			return true;
		if (n < 10)
			return false;
		int sum = 0;
		while (n > 0) {
			int right = n % 10;
			n /= 10;
			sum += right * right;
		}
		return isHappy2(sum);
	}

	public static boolean isHappy(int n) {
		if (n == 1)
			return true;
		Set<Integer> set = new HashSet<>();
		while (n != 1 && !set.contains(n)) {
			set.add(n);
			n = getNext(n);
		}
		System.out.println(n);
		if (n == 1)
			return true;
		return false;
	}

	public static int getNext(int n) {
		int sum = 0;
		while (n > 0) {
			int right = n % 10;
			n /= 10;
			sum += right * right;
		}
		return sum;
	}

	public static int countOccurrence(int[] arr, int value) {
		int i = 0, count = 0, len = arr.length;
		while (i < len) {
			if (arr[i] == value)
				count += 1;
			i++;
		}
		return count;
	}

	public int mincostTickets(int[] days, int[] costs) {
		boolean[] dayIncluded = new boolean[366];
		for (int day : days) {
			dayIncluded[day] = true;
		}
		int[] dp = new int[366];
		for (int i = 1; i < 366; i++) {
			if (!dayIncluded[i]) {
				dp[i] = dp[i - 1];
				continue;
			}
			int min = dp[i - 1] + costs[0];
			min = Math.min(min, dp[Math.max(0, i - 7)] + costs[1]);
			min = Math.min(min, dp[Math.max(0, i - 30)] + costs[2]);
			dp[i] = min;
		}
		return dp[365];
	}

	public static int change(int amount, int[] coins) {
		int[] res = new int[] { 0 };
		changeHelper(amount, coins, 0, res);
		return res[0];
	}

	public static void changeHelper(int amount, int[] coins, int index, int[] res) {
		if (amount == 0) {
			res[0]++;
			return;
		}
		if (amount < 0) {
			return;
		}
		int len = coins.length;
		for (int i = index; i < len; i++) {
			changeHelper(amount - coins[i], coins, i, res);
		}
	}

	public static int coinChange3(int[] coins, int amount) {
		int[] dp = new int[amount];
		int len = coins.length;
		Arrays.sort(coins);
		for (int i = 1; i < amount; i++) {
			int min = Integer.MAX_VALUE;
			for (int j = 0; j < len; j++) {
				int temp = i - coins[j];
				if (temp < 0)
					break;
				if (dp[temp] != -1) {
					if (min > dp[temp]) {
						min = dp[temp];
					}
				}
			}
			dp[i] = min == Integer.MAX_VALUE ? -1 : min + 1;
		}
		return dp[amount - 1];
	}

	public static int coinChange(int[] coins, int amount) {
		return coinChange2(coins, amount, new int[amount]);
	}

	public static int coinChange2(int[] coins, int amount, int[] cache) {
		if (amount == 0) {
			return 0;
		}
		if (amount < 0) {
			return -1;
		}
		if (cache[amount - 1] != 0)
			return cache[amount - 1];
		int len = coins.length;
		int min = Integer.MAX_VALUE;
		for (int i = 0; i < len; i++) {
			int val = coinChange2(coins, amount - coins[i], cache);
			if (val != -1) {
				if (val < min) {
					min = val;
				}
			}
		}
		int res = min == Integer.MAX_VALUE ? -1 : min + 1;
		cache[amount - 1] = res;
		return res;
	}

	public static void coinChange(int[] coins, int amount, int index, int curSum, int count, int[] res) {
		if (curSum == amount) {
			System.out.println(count);
			if (res[0] > count) {
				res[0] = count;
			}
			return;
		}
		if (curSum > amount) {
			return;
		}
		if (index == coins.length)
			return;
		coinChange(coins, amount, index, curSum + coins[index], count + 1, res);
		coinChange(coins, amount, index + 1, curSum, count, res);
	}

	public static int findPrime2(String s, int index, boolean[] prime, Integer[] cache) {
		if (index == s.length()) {
			return 1;
		}
		if (cache[index] != null)
			return cache[index];
		int res = 0;
		for (int i = index + 1; i <= s.length(); i++) {
			String ss = s.substring(index, i);
			if (prime[Integer.valueOf(ss)]) {
				res += findPrime2(s, i, prime, cache);
			}
		}
		cache[index] = res;
		return res;
	}

	public static void findPrime(String s, int index, boolean[] prime, int[] res) {
		if (index == s.length()) {
			res[0]++;
			return;
		}
		for (int i = index + 1; i <= s.length(); i++) {
			String ss = s.substring(index, i);
			if (prime[Integer.valueOf(ss)]) {
				findPrime(s, i, prime, res);
			}
		}
	}

	public static int[] longSub(int[] nums, int k) {
		int firstIndex = 0;
		int len = nums.length;
		for (int i = 0; i < len - k + 1; i++) {
			if (nums[i] > nums[firstIndex])
				firstIndex = i;
		}
		int[] res = new int[k];
		int lastIndex = firstIndex + k - 1;
		for (int i = firstIndex; i <= lastIndex; i++) {
			res[i - firstIndex] = nums[i];
		}
		return res;
	}

	public static void giveMeMaxTime(String time) {
		char[] ch = time.toCharArray();
		if (ch[0] == '?') {
			ch[0] = (ch[1] == '?' || ch[1] <= '3') ? '2' : '1';
		}
		if (ch[1] == '?') {
			ch[1] = (ch[0] == '2') ? '3' : '9';
		}
		if (ch[3] == '?') {
			ch[3] = '5';
		}
		if (ch[4] == '?') {
			ch[4] = '9';
		}
		System.out.println(new String(ch));
	}

	public static int[] compareStrings(String A, String B) {
		String[] strsA = A.split(",");
		String[] strsB = B.split(",");
		int[] freqs = new int[11];
		int[] res = new int[strsB.length];
		for (String s : strsA) {
			char[] ch = s.toCharArray();
			int[] count = new int[26];
			int minIndex = 26;
			for (char c : ch) {
				int tempIndex = c - 'a';
				count[tempIndex]++;
				if (minIndex > tempIndex)
					minIndex = tempIndex;
			}
			int curFreq = count[minIndex];
			freqs[curFreq]++;
		}
//		get the prefix sum
		for (int i = 1; i < 11; i++) {
			freqs[i] += freqs[i - 1];
		}
		for (int i = 0; i < strsB.length; i++) {
			String s = strsB[i];
			int[] count = new int[26];
			int minIndex = 26;
			for (char c : s.toCharArray()) {
				int tempIndex = c - 'a';
				count[tempIndex]++;
				if (minIndex > tempIndex)
					minIndex = tempIndex;
			}
			int curFreq = count[minIndex];
			res[i] = freqs[curFreq - 1];
		}
		return res;
	}

	public static void giveMeMaxTime2(String tim) {
		char[] timChar = tim.toCharArray();

		if (timChar[0] == '?')
			timChar[0] = (timChar[1] <= '3' || timChar[1] == '?') ? '2' : '1';

		if (timChar[1] == '?')
			timChar[1] = (timChar[0] == '2') ? '3' : '9';

		timChar[3] = (timChar[3] == '?') ? '5' : timChar[3];
		timChar[4] = (timChar[4] == '?') ? '9' : timChar[4];
		System.out.println(new String(timChar));

	}

	// Encodes a list of strings to a single string.
	public String encode(List<String> strs) {
		StringBuilder sb = new StringBuilder();
		for (String s : strs) {
			sb.append(s.length()).append('/').append(s);
		}
		return sb.toString();
	}

	// Decodes a single string to a list of strings.
	public List<String> decode(String s) {
		List<String> ret = new ArrayList<String>();
		int i = 0;
		while (i < s.length()) {
			int slash = s.indexOf('/', i);
			int size = Integer.valueOf(s.substring(i, slash));
			i = slash + size + 1;
			ret.add(s.substring(slash + 1, i));
		}
		return ret;
	}

	public void rotate2(int[][] matrix) {
		int len = matrix.length - 1;
		rotate(matrix, 0, len);
	}

	public static void rotate2(int[][] matrix, int index, int len) {
		if (len <= 0)
			return;
		int temp = 0;
		for (int i = 0; i < len; i++) {
			temp = matrix[index][index + i];
			matrix[index][index + i] = matrix[index + len - i][index];
			matrix[index + len - i][index] = matrix[index + len][index + len - i];
			matrix[index + len][index + len - i] = matrix[index + i][index + len];
			matrix[index + i][index + len] = temp;
		}
		rotate2(matrix, index + 1, len - 2);
	}

	public static void mainDia(int[][] matrix) {
		int len = matrix.length;
		int temp = 0;
		for (int i = 0; i < len; i++) {
			for (int j = i + 1; j < len; j++) {
				temp = matrix[i][j];
				matrix[i][j] = matrix[j][i];
				matrix[j][i] = temp;
			}
		}
	}

	public static void antiDia(int[][] matrix) {
		int len = matrix.length;
		int temp = 0;
		for (int i = 0; i < len; i++) {
			for (int j = len - 2 - i; j >= 0; j--) {
				temp = matrix[i][j];
				matrix[i][j] = matrix[len - j - 1][len - i - 1];
				matrix[len - j - 1][len - i - 1] = temp;
			}
		}
	}

	public int minCostII(int[][] costs) {
		if (costs == null || costs.length == 0 || costs[0] == null || costs[0].length == 0)
			return 0;
		int len = costs.length;
		int k = costs[0].length;
		int[][] dp = new int[len][k];
		for (int i = 0; i < k; i++) {
			dp[0][i] = costs[0][i];
		}
		for (int i = 1; i < len; i++) {
			for (int j = 0; j < k; j++) {
				int min = Integer.MAX_VALUE;
				for (int p = 0; p < k; p++) {
					if (p == j) {
						continue;
					} else if (dp[i - 1][p] < min) {
						min = dp[i - 1][p];
					}
				}
				dp[i][j] = costs[i][j] + min;
			}
		}
		int min = Integer.MAX_VALUE;
		for (int i = 0; i < k; i++) {
			if (dp[len - 1][i] < min)
				min = dp[len - 1][i];
		}
		return min;
	}

	public int minCost(int[][] costs) {
		if (costs == null || costs.length == 0 || costs[0] == null || costs[0].length == 0)
			return 0;
		int len = costs.length;
		int[][] dp = new int[len][3];
		dp[0][0] = costs[0][0];
		dp[0][1] = costs[0][1];
		dp[0][2] = costs[0][2];
		for (int i = 1; i < len; i++) {
			dp[i][0] = costs[i][0] + Math.min(dp[i - 1][1], dp[i - 1][2]);
			dp[i][1] = costs[i][1] + Math.min(dp[i - 1][0], dp[i - 1][2]);
			dp[i][2] = costs[i][2] + Math.min(dp[i - 1][0], dp[i - 1][1]);
		}
		return Math.min(dp[len - 1][0], Math.min(dp[len - 1][1], dp[len - 1][2]));
	}

	public static int minCost(int[] cost, int n) {
		int[] dp = new int[n];
		dp[0] = cost[0];
		for (int i = 1; i < n; i++) {
			if (i < 10) {
				dp[i] = cost[i];
			} else {
				dp[i] = Integer.MAX_VALUE;
			}
			for (int j = 0; j <= i / 2; j++) {
				dp[i] = Math.min(dp[j] + dp[i - j - 1], dp[i]);
			}
		}
		return dp[n - 1];
	}

	public static int findLongest(int[] a, int[] b) {
		int[] c = new int[a.length + b.length];
		Set<Integer> set = new HashSet<>();
		merge(a, b, c, set);
		int[] max = new int[] { -1 };
		helper(c, 0, new ArrayList<Integer>(), set, -1, max);
		return max[0];
	}

	public static void merge(int[] a, int[] b, int[] c, Set<Integer> set) {
		int i = 0, j = 0, index = 0;
		int len1 = a.length, len2 = b.length;
		while (i < len1 && j < len2) {
			if (a[i] < b[j]) {
				c[index] = a[i++];
				set.add(index);
			} else {
				c[index] = b[j++];
			}
			index++;
		}
		if (i == len1) {
			while (j < len2) {
				c[index++] = b[j++];
			}
		} else {
			while (i < len1) {
				c[index] = b[i++];
				set.add(index);
				index++;
			}
		}
	}

	public static void helper(int[] c, int i, List<Integer> list, Set<Integer> set, int difference, int[] max) {
		if (i == c.length) {
			max[0] = Math.max(max[0], list.size());
			return;
		}
		int size = list.size();
		if (set.contains(i)) {
			list.add(c[i]);
			if (size == 0) {
				helper(c, i + 1, list, set, difference, max);
			} else {
				int curDiff = c[i] - list.get(size - 1);
				if (difference < 0) {
					helper(c, i + 1, list, set, curDiff, max);
				} else if (difference == curDiff) {
					helper(c, i + 1, list, set, difference, max);
				}
			}
			list.remove(size);
		} else {
			list.add(c[i]);
			if (size == 0) {
				helper(c, i + 1, list, set, difference, max);
			} else {
				int curDiff = c[i] - list.get(size - 1);
				if (difference < 0) {
					helper(c, i + 1, list, set, curDiff, max);
				} else if (difference == curDiff) {
					helper(c, i + 1, list, set, difference, max);
				}
			}
			list.remove(size);
			helper(c, i + 1, list, set, difference, max);
		}

	}

	public static int longestSubstring(String s, int k) {
		if (s == null || s.length() == 0)
			return 0;
		int[] count = new int[26];
		char[] ch = s.toCharArray();
		for (char c : ch) {
			count[c - 'a']++;
		}
		int len = ch.length;
		int start = -1, max = 0;
		for (int i = 0; i < len; i++) {
			int f = count[ch[i] - 'a'];
			if (f >= k && start == -1) {
				start = i;
			}
			if (f < k && start >= 0) {
				max = Math.max(max, longestSubstring(s.substring(start, i), k));
				start = -1;
			}
		}
		if (start == 0) {
			max = len;
		} else if (start > 0) {
			max = Math.max(max, longestSubstring(s.substring(start), k));
		}
		return max;
	}

	public int maxCoins(int[] nums) {
		if (nums == null || nums.length == 0)
			return 0;
		int len = nums.length;
		int[][] dp = new int[len][len];
		for (int i = len - 1; i >= 0; i--) {
			for (int j = i; j < len; j++) {
				int left = i == 0 ? 1 : nums[i - 1];
				int right = j == len - 1 ? 1 : nums[j + 1];
				for (int k = i; k <= j; k++) {
					int a = k == i ? 0 : dp[i][k - 1];
					int b = k == j ? 0 : dp[k + 1][j];
					dp[i][j] = Math.max(dp[i][j], a + b + nums[k] * left * right);
				}
			}
		}
		return dp[0][len - 1];
	}

	public List<Integer> findMinHeightTrees(int n, int[][] edges) {
		class TreeNode {
			int val;
			Set<Integer> nei;

			TreeNode(int val) {
				this.val = val;
				nei = new HashSet<>();
			}

			void addNeibors(int val) {
				nei.add(val);
			}

			void deleteNeibors(Integer val) {
				nei.remove(val);
			}
		}
		List<Integer> res = new ArrayList<>();
		if (n == 1) {
			res.add(0);
			return res;
		}
		TreeNode[] treeNode = new TreeNode[n];
		for (int i = 0; i < n; i++) {
			treeNode[i] = new TreeNode(i);
		}
		for (int[] e : edges) {
			treeNode[e[0]].addNeibors(e[1]);
			treeNode[e[1]].addNeibors(e[0]);
		}
		Queue<Integer> queue = new LinkedList<>();
		int count = n;
		for (int i = 0; i < n; i++) {
			if (treeNode[i].nei.size() == 1) {
				queue.offer(i);
				count--;
			}
		}
		while (count > 0) {
			int size = queue.size();
			while (size-- > 0) {
				int cur = queue.poll();
				for (int i : treeNode[cur].nei) {
					treeNode[i].nei.remove(cur);
					if (treeNode[i].nei.size() == 1) {
						queue.offer(i);
						count--;
					}
				}
			}
		}
		for (int i : queue) {
			res.add(i);
		}
		return res;
	}

	public static void helper(int n, char left, char middle, char right) {
		if (n == 1) {
			move(left, right);
		} else {
			helper(n - 1, left, right, middle);
			move(left, right);
			helper(n - 1, middle, left, right);
		}
	}

	private static void move(char left, char right) {
		System.out.println(left + "->" + right);
	}

	public static boolean PredictTheWinner(int[] nums) {
		int len = nums.length;
		int[][] dp = new int[len][len];
		int sum = nums[len - 1];
		dp[len - 1][len - 1] = nums[len - 1];
		for (int i = len - 2; i >= 0; i--) {
			sum += nums[i];
			dp[i][i] = nums[i];
			dp[i][i + 1] = Math.max(nums[i], nums[i + 1]);
			for (int j = i + 2; j < len; j++) {
				int a = Math.min(dp[i + 2][j], dp[i + 1][j - 1]) + nums[i];
				int b = Math.min(dp[i + 1][j - 1], dp[i][j - 2]) + nums[j];
				dp[i][j] = Math.max(a, b);
			}
		}
		return dp[0][len - 1] >= sum - dp[0][len - 1];
	}

	public boolean isInterleave(String s1, String s2, String s3) {
		char[] ch1 = s1.toCharArray();
		char[] ch2 = s2.toCharArray();
		char[] ch3 = s3.toCharArray();
		int len1 = ch1.length;
		int len2 = ch2.length;
		if (len1 + len2 != ch3.length)
			return false;
		boolean[][] dp = new boolean[len1 + 1][len2 + 1];
		dp[0][0] = true;
		for (int i = 0; i < len1; i++) {
			if (ch1[i] == ch3[i]) {
				dp[i + 1][0] = dp[i][0];
			}
		}
		for (int j = 0; j < len2; j++) {
			if (ch2[j] == ch3[j]) {
				dp[0][j + 1] = dp[0][j];
			}
		}
		for (int i = 0; i < len1; i++) {
			for (int j = 0; j < len2; j++) {
				int k = i + j + 1;
				if (ch1[i] == ch3[k]) {
					dp[i + 1][j + 1] = dp[i][j + 1];
				}
				if (ch2[j] == ch3[k]) {
					dp[i + 1][j + 1] |= dp[i + 1][j];
				}
			}
		}
		return dp[len1][len2];
	}

	public static List<Integer> findMin(int m, int n, int[][] queries) {
		TreeMap<Integer, Integer> map = new TreeMap<Integer, Integer>();
		int[][] matrix = new int[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				matrix[i][j] = (i + 1) * (j + 1);
				map.put(matrix[i][j], map.getOrDefault(matrix[i][j], 0) + 1);
			}
		}
		List<Integer> res = new ArrayList<>();
		for (int[] query : queries) {
			if (query.length == 1) {
				for (int key : map.keySet()) {
					if (map.get(key) != 0) {
						res.add(key);
						break;
					}
				}
			} else {
				if (query[0] == 1) {
					int row = query[1];
					for (int j = 0; j > n; j++) {
						if (matrix[row][j] != -1) {
							map.put(matrix[row][j], map.get(matrix[row][j] - 1));
							matrix[row][j] = -1;
						}
					}
				} else {
					int col = query[1];
					for (int i = 0; i < m; i++) {
						if (matrix[i][col] != -1) {
							map.put(matrix[i][col], map.get(matrix[i][col]) - 1);
							matrix[i][col] = -1;
						}
					}
				}
			}
		}
		return res;
	}

	public static int maxScore2(int n, int[] nums) {
		int[] dp = new int[n];
		for (int i = n - 2; i >= 0; i--) {
			int temp = nums[i];
			int min = n;
			for (int j = 1; j <= temp; j++) {
				if (i + j > n - 1)
					break;
				int val = dp[i + j] + 1;
				if (val < min)
					min = val;
			}
			dp[i] = min;
		}
		return n - dp[0];
	}

	public static int maxScore(int n, int[] nums) {
		int count = 0, i = 0, maxRange = 0;
		while (i <= maxRange) {
			if (maxRange >= n - 1)
				return n - count;
			int temp = i + nums[i];
			if (temp > maxRange) {
				maxRange = temp;
				count++;
			}
			i++;
		}
		return 0;
	}

	public static int longestValidParentheses(String s) {
		char[] ch = s.toCharArray();
		int len = ch.length;
		int[] dp = new int[len + 1];
		int max = 0;
		for (int i = 1; i < len; i++) {
			if (ch[i] == ')') {
				if (ch[i - 1] == '(') {
					dp[i + 1] = dp[i - 1] + 2;
				} else {
					if ((i - dp[i] - 1) >= 0) {
						if (ch[i - dp[i] - 1] == '(') {
							dp[i + 1] = dp[i] + dp[i - dp[i] - 1] + 2;
						}
					}
				}
				max = Math.max(max, dp[i + 1]);
			}
		}
		return max;
	}

	public static int maxCleanRows(int n, int[][] matrix) {
		Set<Integer> set = new HashSet<>();
		for (int i = 0; i < n; i++) {
			set.add(i);
		}
		int[] max = new int[] { 0 };
		helper(n, matrix, 0, set, max);
		return max[0];
	}

	public static void helper(int n, int[][] matrix, int index, Set<Integer> set, int[] max) {
		if (index == n) {
			max[0] = Math.max(max[0], set.size());
			return;
		}
		doNothing(n, index, matrix, set);
		helper(n, matrix, index + 1, set, max);
		doNothingRecover(n, index, matrix, set);
		helper(n, matrix, index + 1, set, max);
	}

	public static void doNothing(int n, int col, int[][] matrix, Set<Integer> set) {
		for (int row = 0; row < n; row++) {
			if (matrix[row][col] == 0) {
				set.remove(row);
			}
		}
	}

	public static void doNothingRecover(int n, int col, int[][] matrix, Set<Integer> set) {
		for (int row = 0; row < n; row++) {
			if (matrix[row][col] == 0) {
				set.add(row);
			} else {
				set.remove(row);
			}
		}
	}

	private static int ribbon(int[] arr, int k) {
		int hi = 0;
		for (int i : arr)
			hi = Math.max(hi, i);
		int lo = 1;
		int res = 0;
		while (lo <= hi) {
			int mid = (lo + hi) / 2;
			int curr = 0;
			for (int i : arr)
				curr += i / mid;
			if (curr >= k) {
				res = Math.max(res, mid);
				lo = mid + 1;
			} else
				hi = mid - 1;

		}
		return res;
	}

	public static void helper(int n, int[] list) {
		int[][] matrix = new int[n][n];
		int index = 0;
		for (int i = n - 1; i >= 0; i--) {
			int num = n - i;
			int x = n - 1, y = i;
//			System.out.println(num);
			while (num-- > 0) {
				matrix[x][y] = list[index++];
				x -= 1;
				y += 1;
			}
		}
		for (int i = n - 2; i >= 0; i--) {
			int num = i + 1;
			int x = i, y = 0;
			while (num-- > 0) {
				matrix[x][y] = list[index++];
				x -= 1;
				y += 1;
			}
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; i++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
	}

	public static String sumOfString(String s1, String s2) {
		if (s1 == null || s1.length() == 0)
			return s2;
		if (s2 == null || s2.length() == 0)
			return s1;
		int len1 = s1.length();
		int len2 = s2.length();
		StringBuilder sb = new StringBuilder();
		int idx1 = len1 - 1;
		int idx2 = len2 - 1;
		while (idx1 >= 0 && idx2 >= 0) {
			char c1 = s1.charAt(idx1--);
			char c2 = s2.charAt(idx2--);
			int num1 = c1 - '0';
			int num2 = c2 - '0';
			int sum = num1 + num2;
			sb.insert(0, Integer.toString(sum));
		}

		while (idx1 >= 0) {
			sb.insert(0, s1.charAt(idx1--));
		}

		while (idx2 >= 0) {
			sb.insert(0, s2.charAt(idx2--));
		}

		return sb.toString();
	}

	public List<List<Integer>> queensAttacktheKing2(int[][] queens, int[] king) {
		List<List<Integer>> res = new ArrayList<>();
		boolean[][] matrix = new boolean[8][8];
		for (int[] i : queens) {
			matrix[i[0]][i[1]] = true;
		}
		int[][] directions = { { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { 1, 0 }, { 1, -1 } };
		for (int[] dir : directions) {
			int x = dir[0], y = dir[1];
			int curX = king[0], curY = king[1];
			while (curX >= 0 && curX < 8 && curY >= 0 && curY < 8) {
				if (matrix[curX][curY]) {
					res.add(Arrays.asList(curX, curY));
					break;
				}
				curX += x;
				curY += y;
			}
		}
		return res;
	}

	public int minimumTotal(List<List<Integer>> triangle) {
		int row = triangle.size();
		int col = triangle.get(row - 1).size();
		int[][] dp = new int[row][col];
		for (int i = 0; i < col; i++) {
			dp[row - 1][i] = triangle.get(row - 1).get(i);
		}
		for (int i = row - 2; i >= 0; i--) {
			for (int j = 0; j <= i; j++) {
				dp[i][j] = Math.min(dp[i + 1][j], dp[i + 1][j + 1]) + triangle.get(i).get(j);
			}
		}
		return dp[0][0];
	}

	public static List<Integer> getRow(int rowIndex) {
		List<Integer> res = new ArrayList<>();
		res.add(1);
		if (rowIndex == 0)
			return res;
		for (int i = 1; i <= rowIndex; i++) {
			List<Integer> cur = new ArrayList<>();
			cur.add(1);
			for (int j = 0; j < i - 1; j++) {
				cur.add(res.get(j) + res.get(j + 1));
			}
			cur.add(1);
			res = cur;
		}
		return res;
	}

	public static List<List<Integer>> generate(int numRows) {
		List<List<Integer>> res = new ArrayList<>();
		if (numRows == 0)
			return res;
		List<Integer> first = new ArrayList<>();
		first.add(1);
		res.add(first);
		for (int i = 1; i < numRows; i++) {
			List<Integer> before = res.get(i - 1);
			List<Integer> cur = new ArrayList<>();
			cur.add(1);
			for (int j = 0; j < i - 1; j++) {
				cur.add(before.get(j) + before.get(j + 1));
			}
			cur.add(1);
			res.add(cur);
		}
		return res;
	}

	public static int minCut2(String s) {
		if (s == null || s.length() == 0)
			return 0;
		char[] ch = s.toCharArray();
		int len = ch.length;
		boolean[][] isPalindrome = new boolean[len][len];
		int[] dp = new int[len + 1];
		for (int i = len; i >= 0; i--) {
			dp[i] = len - 1 - i;
			for (int j = i; j < len; j++) {
				if (i == j || (ch[i] == ch[j] && (i == j - 1 || isPalindrome[i + 1][j - 1]))) {
					dp[i] = Math.min(dp[i], dp[j + 1] + 1);
					isPalindrome[i][j] = true;
				}
			}
		}
		return dp[0];
	}

	public static List<List<String>> partition2(String s) {
		List<List<String>> res = new ArrayList<>();
		if (s == null || s.length() == 0)
			return res;
		char[] ch = s.toCharArray();
		int len = ch.length;
		boolean[][] dp = new boolean[len][len];
		for (int i = len - 1; i >= 0; i--) {
			for (int j = i; j < len; j++) {
				dp[i][j] = i == j || (ch[i] == ch[j] && (i + 1 == j || dp[i + 1][j - 1]));
			}
		}
		Map<Integer, List<List<String>>> tmpRes = new HashMap<>();
		partitionHelper2(s, 0, 0, dp, tmpRes, new ArrayList<String>(), res);
		return res;
	}

	public static void partitionHelper2(String s, int l, int r, boolean[][] isPalindrome,
			Map<Integer, List<List<String>>> tmpRes, List<String> cur, List<List<String>> res) {
		int len = s.length();
		if (tmpRes.containsKey(l)) {
			List<List<String>> curRes = tmpRes.get(l);
			for (List<String> ll : curRes) {
				List<String> newL = new ArrayList<>(cur);
				newL.addAll(ll);
				res.add(newL);
			}
			return;
		}
		if (l >= len) {
			res.add(new ArrayList<String>(cur));
			return;
		}
		Map<Integer, List<List<String>>> now = new HashMap<>();
		for (int i = r; i < len; i++) {
			if (isPalindrome[l][i]) {
				List<List<String>> lili = now.getOrDefault(l, new ArrayList<List<String>>());
				String tmpS = s.substring(l, i + 1);
				cur.add(tmpS);
				partitionHelper2(s, i + 1, i + 1, isPalindrome, tmpRes, cur, res);
				if (tmpRes.containsKey(i + 1)) {
					List<String> li = new ArrayList<>();
					li.add(tmpS);
					List<List<String>> curII = tmpRes.get(i + 1);
					for (List<String> ll : curII) {
						List<String> newI = new ArrayList<>(li);
						newI.addAll(ll);
						lili.add(newI);
					}
				}
				cur.remove(cur.size() - 1);
			}
		}
	}

	public int minCut(String s) {
		char[] ch = s.toCharArray();
		int len = ch.length;
		boolean[][] dp = new boolean[len][len];
		for (int i = len - 1; i >= 0; i--) {
			for (int j = i; j < len; j++) {
				dp[i][j] = i == j || (ch[i] == ch[j] && (i + 1 == j || dp[i + 1][j - 1]));
			}
		}
		int[] min = new int[] { len };
		int[] map = new int[len + 1];
		Arrays.fill(map, len);
		map[len] = 0;
		minCutHelper(s, 0, 0, dp, map, -1, min);
		return min[0];
	}

	public static void minCutHelper(String s, int l, int r, boolean[][] isPalindrome, int[] map, int cur, int[] min) {
		int len = s.length();
		if (map[l] != len) {
			min[0] = Math.min(min[0], map[l] + cur);
			return;
		}
		for (int i = r; i < len; i++) {
			if (isPalindrome[l][i]) {
				minCutHelper(s, i + 1, i + 1, isPalindrome, map, cur + 1, min);
				map[l] = Math.min(map[l], map[i + 1] + 1);
			}
		}
	}

	public static List<List<String>> partition(String s) {
		List<List<String>> res = new ArrayList<>();
		if (s == null || s.length() == 0)
			return res;
		char[] ch = s.toCharArray();
		int len = ch.length;
		boolean[][] dp = new boolean[len][len];
		for (int i = len - 1; i >= 0; i--) {
			for (int j = i; j < len; j++) {
				dp[i][j] = i == j || (ch[i] == ch[j] && (i + 1 == j || dp[i + 1][j - 1]));
			}
		}
		partitionHelper(s, 0, 0, dp, new ArrayList<String>(), res);
		return res;
	}

//	can use memorization
	public static void partitionHelper(String s, int l, int r, boolean[][] isPalindrome, List<String> cur,
			List<List<String>> res) {
		int len = s.length();
		if (l >= len) {
			res.add(new ArrayList<String>(cur));
			return;
		}
		for (int i = r; i < len; i++) {
			if (isPalindrome[l][i]) {
				cur.add(s.substring(l, i + 1));
				partitionHelper(s, i + 1, i + 1, isPalindrome, cur, res);
				cur.remove(cur.size() - 1);
			}
		}
	}

	public void rotate(int[] nums, int k) {
		int len = nums.length;
		k = k % len;
		int[] tmp = new int[k];
		for (int i = len - k; i < len; i++) {
			tmp[i - len + k] = nums[i];
		}
		for (int i = len - 1 - k; i >= 0; i--) {
			nums[i + k] = nums[i];
		}
		for (int i = 0; i < k; i++) {
			nums[i] = tmp[i];
		}
	}

	public void reverseWords(char[] s) {
		if (s == null || s.length == 0)
			return;
		int len = s.length;
		reverse(s, 0, len - 1);
		int start = 0, end = 0;
		for (int i = 0; i < len; i++) {
			if (s[i] != ' ' && (i == len - 1 || s[i + 1] == ' ')) {
				end = i;
				reverse(s, start, end);
				start = i + 2;
			}
		}
	}

	public void reverse(char[] s, int left, int right) {
		while (left < right) {
			char tmp = s[left];
			s[left] = s[right];
			s[right] = tmp;
			left++;
			right--;
		}
	}

	public static String reverseWords(String s) {
		String[] tmp = s.split(" ");
		int len = tmp.length;
		StringBuilder sb = new StringBuilder();
		for (int i = len - 1; i >= 0; i--) {
			if (tmp[i].equals(""))
				continue;
			sb.append(tmp[i]).append(" ");
		}
		return sb.toString().trim();
	}

	public static List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
		List<List<Integer>> left = new ArrayList<>();
		List<List<Integer>> right = new ArrayList<>();
		List<List<Integer>> up = new ArrayList<>();
		List<List<Integer>> down = new ArrayList<>();
		List<List<Integer>> lu = new ArrayList<>();
		List<List<Integer>> ld = new ArrayList<>();
		List<List<Integer>> ru = new ArrayList<>();
		List<List<Integer>> rd = new ArrayList<>();
		int x = king[0], y = king[1];
		for (int[] q : queens) {
			int row = q[0];
			int col = q[1];
			List<Integer> cur = new ArrayList<>();
			cur.add(row);
			cur.add(col);
			if (row == x) {
				if (col > y) {
					right.add(cur);
				} else {
					left.add(cur);
				}
			} else if (col == y) {
				if (row > x) {
					down.add(cur);
				} else {
					up.add(cur);
				}
			} else if (row > x && col > y) {
				if (row - x == col - y) {
					rd.add(cur);
				}
			} else if (row > x && col < y) {
				if (row - x == y - col) {
					ld.add(cur);
				}
			} else if (row < x && col > y) {
				if (x - row == col - y) {
					ru.add(cur);
				}
			} else {
				if (x - row == y - col) {
					lu.add(cur);
				}
			}
		}
		Collections.sort(left, new Comparator<List<Integer>>() {
			public int compare(List<Integer> o1, List<Integer> o2) {
				return o2.get(1) - o1.get(1);
			}
		});
		Collections.sort(right, new Comparator<List<Integer>>() {
			public int compare(List<Integer> o1, List<Integer> o2) {
				return o1.get(1) - o2.get(1);
			}
		});
		Collections.sort(up, new Comparator<List<Integer>>() {
			public int compare(List<Integer> o1, List<Integer> o2) {
				return o2.get(0) - o1.get(0);
			}
		});
		Collections.sort(down, new Comparator<List<Integer>>() {
			public int compare(List<Integer> o1, List<Integer> o2) {
				return o1.get(0) - o2.get(0);
			}
		});
		Collections.sort(lu, new Comparator<List<Integer>>() {
			public int compare(List<Integer> o1, List<Integer> o2) {
				return o2.get(0) - o1.get(0);
			}
		});
		Collections.sort(ru, new Comparator<List<Integer>>() {
			public int compare(List<Integer> o1, List<Integer> o2) {
				return o2.get(0) - o1.get(0);
			}
		});
		Collections.sort(ld, new Comparator<List<Integer>>() {
			public int compare(List<Integer> o1, List<Integer> o2) {
				return o1.get(0) - o2.get(0);
			}
		});
		Collections.sort(rd, new Comparator<List<Integer>>() {
			public int compare(List<Integer> o1, List<Integer> o2) {
				return o1.get(0) - o2.get(0);
			}
		});
		List<List<Integer>> res = new ArrayList<>();
		if (left.size() > 0) {
			res.add(left.get(0));
		}
		if (right.size() > 0) {
			res.add(right.get(0));
		}
		if (up.size() > 0) {
			res.add(up.get(0));
		}
		if (down.size() > 0) {
			res.add(down.get(0));
		}
		if (lu.size() > 0) {
			res.add(lu.get(0));
		}
		if (ld.size() > 0) {
			res.add(ld.get(0));
		}
		if (ru.size() > 0) {
			res.add(ru.get(0));
		}
		if (rd.size() > 0) {
			res.add(rd.get(0));
		}
		return res;
	}

	public static int balancedStringSplit(String s) {
		char[] ch = s.toCharArray();
		int l = 0, r = 0, len = ch.length;
		int count = 0;
		for (int i = 0; i < len; i++) {
			if (ch[i] == 'L') {
				l++;
			} else {
				r++;
			}
			if (l == r) {
				l = 0;
				r = 0;
				count++;
			}
		}
		return count;
	}

	public void buildPrefix() {
		int MAX = 1000;
		int[] prefix = new int[MAX + 1];
		boolean prime[] = new boolean[MAX + 1];
		Arrays.fill(prime, true);
		for (int p = 2; p * p <= MAX; p++) {
			if (prime[p] == true) {
				for (int i = p * 2; i <= MAX; i += p)
					prime[i] = false;
			}
		}
		prefix[0] = prefix[1] = 0;
		for (int p = 2; p <= MAX; p++) {
			prefix[p] = prefix[p - 1];
			if (prime[p])
				prefix[p]++;
		}
	}

	public static int maxWeight(String str, int p, int s, int t) {
		char[] ch = str.toCharArray();
		int len = str.length();
		int[] dp = new int[len + 1];
		dp[0] = 0;
		dp[1] = s;
		for (int i = 2; i <= len; i++) {
			if (ch[i - 1] != ch[i - 2]) {
				dp[i] = Math.max(p + dp[i - 2], dp[i - 1] + s);
			} else if (ch[i - 2] != ch[i - 3]) {
				if (i > 2) {
					dp[i] = Math.max(dp[i - 1] + s - t, Math.max(dp[i - 2] + p - t, dp[i - 1] + s));
				} else {
					dp[i] = Math.max(dp[i - 1] + s - t, dp[i - 2] + p - t);
				}
			}
		}
		return dp[len];
	}

	public int numDistinct(String s, String t) {
		char[] cs = s.toCharArray(), ct = t.toCharArray();
		int len1 = cs.length, len2 = ct.length;
		int[][] dp = new int[len1 + 1][len2 + 1];
		for (int i = 0; i <= len1; i++) {
			dp[i][0] = 1;
		}
		for (int j = 1; j <= len2; j++) {
			dp[0][j] = 0;
		}
		for (int i = 1; i <= len1; i++) {
			char a = cs[i - 1];
			for (int j = 1; j <= len2; j++) {
				char b = ct[j - 1];
				if (a != b) {
					dp[i][j] = dp[i - 1][j];
				} else {
					dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
				}
			}
		}
		return dp[len1][len2];
	}

	public int minDistance(String word1, String word2) {
		char[] w1 = word1.toCharArray(), w2 = word2.toCharArray();
		int len1 = w1.length, len2 = w2.length;
		int[][] dp = new int[len1 + 1][len2 + 1];
		for (int i = 0; i <= len1; i++) {
			dp[i][0] = i;
		}
		for (int j = 0; j <= len2; j++) {
			dp[0][j] = j;
		}
		for (int i = 1; i <= len1; i++) {
			char a = w1[i - 1];
			for (int j = 1; j <= len2; j++) {
				char b = w2[j - 1];
				if (a == b) {
					dp[i][j] = dp[i - 1][j - 1];
				} else {
					dp[i][j] = Math.min(dp[i][j - 1], Math.min(dp[i - 1][j], dp[i - 1][j - 1])) + 1;
				}
			}
		}
		return dp[len1][len2];
	}

	private static int helper(int money) {
		int len = money / 50;
		int[] dp = new int[len + 1];
		dp[0] = 0;
		dp[1] = 50;
		dp[2] = 100;
		dp[3] = 0;
		dp[4] = 0;
		for (int i = 5; i <= len; i++) {
			dp[i] = Math.min(dp[i - 3], dp[i - 4]);
		}
		return dp[len];
	}

	public static String findShortestWay(int[][] maze, int[] ball, int[] hole) {
		class Path {
			int[] val;
			StringBuilder sb;

			Path(int[] val, StringBuilder sb) {
				this.val = val;
				this.sb = sb;
			}
		}
		int row = maze.length, col = maze[0].length;
		int[][] distance = new int[row][col];
		for (int[] r : distance)
			Arrays.fill(r, Integer.MAX_VALUE);
		distance[ball[0]][ball[1]] = 0;
		maze[hole[0]][hole[1]] = 1;
		PriorityQueue<Path> pq = new PriorityQueue<>(new Comparator<Path>() {
			@Override
			public int compare(Path o1, Path o2) {
				if (o1.val[2] == o2.val[2]) {
					return o1.sb.toString().compareTo(o2.sb.toString());
				} else {
					return o1.val[2] - o2.val[2];
				}
			}
		});
		pq.offer(new Path(new int[] { ball[0], ball[1], 0 }, new StringBuilder()));
		int[][] directions = { { 1, 0 }, { 0, -1 }, { 0, 1 }, { -1, 0 } };
		while (!pq.isEmpty()) {
			Path cur = pq.poll();
			int[] curV = cur.val;
			int curX = curV[0], curY = curV[1], curD = curV[2];
			StringBuilder curS = cur.sb;
			if (curD > distance[curX][curY])
				continue;
			if (curX == hole[0] && curY == hole[1])
				return curS.toString();
			for (int[] dir : directions) {
				String s = "";
				switch (dir[0] * 10 + dir[1]) {
				case 10:
					s = "d";
					break;
				case 1:
					s = "r";
					break;
				case -1:
					s = "l";
					break;
				case -10:
					s = "u";
					break;
				}
				int count = 0;
				int xx = curX, yy = curY;
				int x = dir[0], y = dir[1];
				while (xx >= 0 && xx < row && yy >= 0 && yy < col && maze[xx][yy] == 0) {
					xx += x;
					yy += y;
					count++;
				}
				if (xx != hole[0] || yy != hole[1]) {
					xx -= x;
					yy -= y;
					count--;
				}
				if (xx == curX && yy == curY)
					continue;
				int tempD = curD + count;
				if (tempD <= distance[xx][yy]) {
					distance[xx][yy] = tempD;
					pq.offer(new Path(new int[] { xx, yy, tempD }, new StringBuilder(curS).append(s)));
				}
			}
		}
		return "impossible";
	}

	public static int shortestDistance(int[][] maze, int[] start, int[] destination) {
		int row = maze.length, col = maze[0].length;
		int[][] distance = new int[row][col];
		for (int[] r : distance)
			Arrays.fill(r, Integer.MAX_VALUE);
		distance[start[0]][start[1]] = 0;
		PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[2] - b[2]);
		pq.offer(new int[] { start[0], start[1], 0 });
		int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };
		while (!pq.isEmpty()) {
			int[] cur = pq.poll();
			int curX = cur[0], curY = cur[1], curD = cur[2];
			if (curD > distance[curX][curY])
				continue;
			if (curX == destination[0] && curY == destination[1])
				return curD;
			for (int[] dir : directions) {
				int count = 0;
				int xx = curX, yy = curY;
				int x = dir[0], y = dir[1];
				while (xx >= 0 && xx < row && yy >= 0 && yy < col && maze[xx][yy] == 0) {
					xx += x;
					yy += y;
					count++;
				}
				xx -= x;
				yy -= y;
				count--;
				int tempD = curD + count;
				if (tempD < distance[xx][yy]) {
					distance[xx][yy] = tempD;
					pq.offer(new int[] { xx, yy, tempD });
				}
			}
		}
		return -1;
	}

	private static void helper(int[][] maze, int x, int y, int[] destination, int[][] directions, int[] record) {
		System.out.println(record[0]);
		if (x == destination[0] && y == destination[1]) {
			record[1] = Math.min(record[0], record[1]);
			return;
		}
		if (maze[x][y] == 1 || maze[x][y] == -1)
			return;
		maze[x][y] = -1;
		for (int[] dir : directions) {
			int xx = dir[0], yy = dir[1];
			int newX = x, newY = y;
			int count = 0;
			while (newX >= 0 && newX < maze.length && newY >= 0 && newY < maze[0].length && maze[newX][newY] != 1) {
				newX += xx;
				newY += yy;
				count++;
			}
			count--;
			int curX = newX - xx, curY = newY - yy;
			record[0] += count;
			helper(maze, curX, curY, destination, directions, record);
			record[0] -= count;
		}
		maze[x][y] = 0;
	}

	public boolean hasPath(int[][] maze, int[] start, int[] destination) {
		return helper(maze, start[0], start[1], destination, new int[][] { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } });
	}

	private boolean helper(int[][] maze, int x, int y, int[] destination, int[][] directions) {
		if (x == destination[0] && y == destination[1]) {
			return true;
		}
		if (maze[x][y] == 1 || maze[x][y] == -1)
			return false;
		maze[x][y] = -1;
		for (int[] dir : directions) {
			int xx = dir[0], yy = dir[1];
			int newX = x, newY = y;
			while (newX >= 0 && newX < maze.length && newY >= 0 && newY < maze[0].length && maze[newX][newY] != 1) {
				newX += xx;
				newY += yy;
			}
			int curX = newX - xx, curY = newY - yy;
			if (helper(maze, curX, curY, destination, directions))
				return true;
		}
		return false;
	}

	public int helper(int n, int coin1, int coin2, int coin3) {
		int min = Integer.MAX_VALUE;
		int first = n / coin1;
		int curVal, second, res;
		for (int i = 0; i <= first; i++) {
			curVal = n - i * coin1;
			second = curVal / coin2;
			for (int j = 0; j <= second; j++) {
				curVal -= coin2;
				res = curVal - curVal / coin3 * coin3;
				if (min > res)
					min = res;
			}
		}
		return min;
	}

	public List<String> findStrobogrammatic(int n) {
		List<String> res = new ArrayList<>();
		if (n == 0)
			return res;
		if (n == 1) {
			res.add("0");
			res.add("1");
			res.add("8");
			return res;
		}
		Map<Character, Character> map = new HashMap<>();
		map.put('6', '9');
		map.put('9', '6');
		map.put('8', '8');
		map.put('1', '1');
		map.put('0', '0');
		char[] ch = new char[n];
		for (char c : map.keySet()) {
			if (c != '0') {
				ch[0] = c;
				ch[n - 1] = map.get(c);
				helper(res, map, 1, n - 2, ch);
			}
		}
		return res;
	}

	private void helper(List<String> res, Map<Character, Character> map, int left, int right, char[] ch) {
		if (left == right) {
			ch[left] = '0';
			res.add(new String(ch));
			ch[left] = '1';
			res.add(new String(ch));
			ch[left] = '8';
			res.add(new String(ch));
			return;
		}
		if (left > right) {
			res.add(new String(ch));
			return;
		}
		for (char c : map.keySet()) {
			ch[left] = c;
			ch[right] = map.get(c);
			helper(res, map, left + 1, right - 1, ch);
		}
	}

	public boolean isStrobogrammatic(String num) {
		Map<Character, Character> map = new HashMap<>();
		map.put('6', '9');
		map.put('9', '6');
		map.put('8', '8');
		map.put('1', '1');
		map.put('0', '0');
		char[] ch = num.toCharArray();
		int left = 0, right = num.length() - 1;
		while (left <= right) {
			if (map.containsKey(ch[left]) && map.get(ch[left]) == ch[right]) {
				left++;
				right--;
			} else {
				return false;
			}
		}
		return true;
	}

	public static boolean isValid2(String s) {
		if (s.length() != 5)
			return false;
		int[] count = new int[26];
		for (int i = 0; i < 5; i++) {
			count[s.charAt(i) - 'a']++;
		}
		for (int i = 0; i < 26; i++) {
			if (count[i] > 1)
				return false;
		}
		return true;
	}

	public int eraseOverlapIntervals(int[][] points) {
		if (points == null || points.length == 0 || points[0] == null || points[0].length == 0)
			return 0;
		Arrays.sort(points, new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				return o1[1] - o2[1];
			}
		});
		int endTime = points[0][1];
		int arrow = 1;
		for (int[] i : points) {
			if (i[0] >= endTime) {
				arrow++;
				endTime = i[1];
			}
		}
		return points.length - arrow;
	}

	public int findMinArrowShots(int[][] points) {
		if (points == null || points.length == 0 || points[0] == null || points[0].length == 0)
			return 0;
		Arrays.sort(points, new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				return o1[1] - o2[1];
			}
		});
		int endTime = points[0][1];
		int arrow = 1;
		for (int[] i : points) {
			if (i[0] > endTime) {
				arrow++;
				endTime = i[1];
			}
		}
		return arrow;
	}

	public int minMeetingRooms(int[][] intervals) {
		int[] start = new int[intervals.length];
		int[] end = new int[intervals.length];
		for (int i = 0; i < intervals.length; i++) {
			start[i] = intervals[i][0];
			end[i] = intervals[i][1];
		}
		Arrays.sort(start);
		Arrays.sort(end);
		int count = 0;
		int endPoint = 0;
		for (int i = 0; i < intervals.length; i++) {
			if (start[i] < end[endPoint]) {
				count++;
			} else {
				endPoint++;
			}
		}
		return count;
	}

	public double[] medianSlidingWindow2(int[] nums, int k) {
		int n = nums.length;
		double[] res = new double[n - k + 1];
		TreeMap<Integer, Integer> maxHeap = new TreeMap<>(Collections.reverseOrder());
		TreeMap<Integer, Integer> minHeap = new TreeMap<>();
		int maxSize = 0;
		int minSize = 0;

		for (int i = 0; i < n; i++) {
			// add
			if (maxSize <= minSize) {
				// add max
				add(minHeap, nums[i]);
				add(maxHeap, remove(minHeap, minHeap.firstKey()));
				maxSize++;
			} else {
				// add min
				add(maxHeap, nums[i]);
				add(minHeap, remove(maxHeap, maxHeap.firstKey()));
				minSize++;
			}

			// calculate
			if (i >= k - 1) {
				// get the median
				res[i - k + 1] = (minSize == maxSize) ? ((long) maxHeap.firstKey() + minHeap.firstKey()) / 2.0
						: maxHeap.firstKey();

				// remove the element which is out of the window
				if (maxHeap.containsKey(nums[i - k + 1])) {
					remove(maxHeap, nums[i - k + 1]);
					maxSize--;
				} else {
					remove(minHeap, nums[i - k + 1]);
					minSize--;
				}
			}

		}

		return res;
	}

	private int remove(TreeMap<Integer, Integer> treeMap, int key) {
		if (treeMap.get(key) == 1) {
			treeMap.remove(key);
		} else {
			treeMap.put(key, treeMap.get(key) - 1);
		}

		return key;
	}

	private void add(TreeMap<Integer, Integer> treeMap, int key) {
		treeMap.put(key, treeMap.getOrDefault(key, 0) + 1);
	}

	private static int helper22(int[][] matrix) {
		int row = matrix.length, col = matrix[0].length;
		int res = 0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int[] max = { 0 };
				dfs(matrix, 0, 0, max);
				res = Math.max(res, max[0]);
			}
		}
		return res;
	}

	private static void dfs(int[][] matrix, int i, int j, int[] max) {
		if (i < 0 || i >= matrix.length || j < 0 || j >= matrix[0].length || matrix[i][j] == 0)
			return;
		max[0] += matrix[i][j];
		matrix[i][j] = 0;
		dfs(matrix, i - 1, j, max);
		dfs(matrix, i + 1, j, max);
		dfs(matrix, i, j + 1, max);
		dfs(matrix, i, j - 1, max);
	}

	public int longestIncreasingPath2(int[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			return 0;
		int row = matrix.length, col = matrix[0].length;
		int[][] field = new int[row][col];
		int max = -1;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (field[i][j] == 0) {
					max = Math.max(longestIncreasingPathHelper(matrix, i, j, Integer.MIN_VALUE, field), max);
				}
			}
		}
		return max;
	}

	private int longestIncreasingPathHelper(int[][] matrix, int i, int j, int prev, int[][] field) {
		if (i < 0 || i >= matrix.length || j < 0 || j >= matrix[0].length || prev >= matrix[i][j])
			return 0;
		if (field[i][j] > 0)
			return field[i][j];
		int res = Math.max(longestIncreasingPathHelper(matrix, i - 1, j, matrix[i][j], field),
				Math.max(longestIncreasingPathHelper(matrix, i + 1, j, matrix[i][j], field),
						Math.max(longestIncreasingPathHelper(matrix, i, j - 1, matrix[i][j], field),
								longestIncreasingPathHelper(matrix, i, j + 1, matrix[i][j], field))))
				+ 1;
		field[i][j] = res;
		return res;
	}

	public static String removeDuplicates2(String s, int k) {
		char[] ch = s.toCharArray();
		char[] cs = new char[ch.length];
		int[] cst = new int[ch.length];
		int p = 0;
		for (char c : ch) {
			if (p > 0 && cs[p - 1] == c) {
				if (++cst[p - 1] == k) {
					p--;
				}
			} else {
				cst[p] = 1;
				cs[p] = c;
				p++;
			}
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < p; i++) {
			for (int j = 0; j < cst[i]; j++) {
				sb.append(cs[i]);
			}
		}
		return sb.toString();
	}

	public static String removeDuplicates(String s, int k) {
		int len = s.length();
		for (int i = 0; i < len; i++) {
			int count = 0, j = i;
			while (j < len && s.charAt(j) == s.charAt(i)) {

				if (++count == k) {
					String newS = helper(s, i - 1, j + 1, k);
					return removeDuplicates(newS, k);
				}
				j++;
			}
		}
		return s;
	}

	public static String helper(String board, int left, int right, int k) {
		int len = board.length();
		while (left >= 0 && right < len) {
			char c = board.charAt(left);
			int count = 0;
			int l = left;
			while (l >= 0 && board.charAt(l) == c) {
				count++;
				l--;
			}

			int r = right;
			while (r < len && board.charAt(r) == c) {
				r++;
				if (++count == k) {
					left = l;
					right = r;
					break;
				}
			}
			break;
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i <= left; i++) {
			sb.append(board.charAt(i));
		}
		for (int i = right; i < len; i++) {
			sb.append(board.charAt(i));
		}
		return sb.toString();
	}

	public static int equalSubstring(String s, String t, int maxCost) {
		char[] ss = s.toCharArray();
		char[] tt = t.toCharArray();
		int slow = 0;
		int cost = 0;
		int maxLen = 0;
		for (int fast = 0; fast < ss.length; fast++) {
			cost += Math.abs(tt[fast] - ss[fast]);
			if (cost > maxCost) {
				cost -= Math.abs(tt[slow] - ss[slow]);
				slow++;
			} else {
				maxLen = Math.max(maxLen, fast - slow + 1);
			}
		}
		return maxLen;
	}

	public boolean uniqueOccurrences(int[] arr) {
		Map<Integer, Integer> map = new HashMap<>();
		for (int i : arr) {
			map.put(i, map.getOrDefault(i, 0) + 1);
		}
		Set<Integer> set = new HashSet<>();
		for (int i : map.keySet()) {
			if (!set.add(map.get(i))) {
				return false;
			}
		}
		return true;
	}

	public double[] medianSlidingWindow(int[] nums, int k) {
		double[] res = new double[nums.length - k + 1];
		PriorityQueue<Integer> small = new PriorityQueue<>(Collections.reverseOrder());
		PriorityQueue<Integer> big = new PriorityQueue<>();
		int count = 0;
		for (int i = 0; i < k; i++) {
			if (small.size() == big.size()) {
				if (small.isEmpty()) {
					small.offer(nums[i]);
				} else if (nums[i] <= big.peek()) {
					small.offer(nums[i]);
				} else {
					small.offer(big.poll());
					big.offer(nums[i]);
				}
			} else {
				if (nums[i] >= small.peek()) {
					big.offer(nums[i]);
				} else {
					big.offer(small.poll());
					small.offer(nums[i]);
				}
			}
		}
		boolean even = k % 2 == 0;
		res[0] = even ? (small.peek() + (double) ((long) big.peek() - small.peek()) / 2) : small.peek();
		for (int i = k; i < nums.length; i++) {
			if (!small.remove(nums[i - k])) {
				big.remove(nums[i - k]);
			}
			if (small.size() <= big.size()) {
				if (small.isEmpty()) {
					small.offer(nums[i]);
				} else if (nums[i] <= big.peek()) {
					small.offer(nums[i]);
				} else {
					small.offer(big.poll());
					big.offer(nums[i]);
				}
			} else {
				if (nums[i] >= small.peek()) {
					big.offer(nums[i]);
				} else {
					big.offer(small.poll());
					small.offer(nums[i]);
				}
			}
			res[i - k + 1] = even ? (small.peek() + (double) ((long) big.peek() - small.peek()) / 2) : small.peek();
		}
		return res;
	}

	public List<Integer> findAnagrams2(String s, String p) {
		List<Integer> res = new ArrayList<>();
		if (s.length() < p.length())
			return res;
		int[] arr = new int[26];
		int count = p.length();
		char[] chP = p.toCharArray();
		char[] chS = s.toCharArray();
		for (char c : chP) {
			arr[c - 'a']++;
		}
		for (int i = 0; i < chP.length; i++) {
			if (arr[chS[i] - 'a']-- > 0) {
				count--;
			}
		}
		if (count == 0)
			res.add(0);
		for (int i = chP.length; i < chS.length; i++) {
			int leftIndex = i - chP.length;
			if (arr[chS[i] - 'a']-- > 0) {
				count--;
			}
			if (arr[chS[leftIndex] - 'a']++ >= 0) {
				count++;
			}
			if (count == 0)
				res.add(leftIndex + 1);
		}
		return res;
	}

	public boolean checkInclusion2(String s1, String s2) {
		if (s2.length() < s1.length())
			return false;
		int[] arr = new int[26];
		int count = s1.length();
		char[] ss1 = s1.toCharArray();
		char[] ss2 = s2.toCharArray();
		for (char c : ss1) {
			arr[c - 'a']++;
		}
		for (int i = 0; i < ss1.length; i++) {
			if (arr[ss2[i] - 'a']-- > 0) {
				count--;
			}
		}
		if (count == 0)
			return true;
		for (int i = s1.length(); i < s2.length(); i++) {
			if (arr[ss2[i] - 'a']-- > 0) {
				count--;
			}
			if (arr[ss2[i - s1.length()] - 'a']++ >= 0) {
				count++;
			}
			if (count == 0)
				return true;
		}
		return false;
	}

	public boolean checkInclusion(String s1, String s2) {
		Map<Character, Integer> map = new HashMap<>();
		char[] chS = s2.toCharArray();
		char[] chP = s1.toCharArray();
		int pLen = chP.length;
		for (Character c : chP) {
			map.put(c, map.getOrDefault(c, 0) + 1);
		}
		int count = 0;
		for (int i = 0; i < chS.length; i++) {
			if (map.containsKey(chS[i])) {
				map.put(chS[i], map.get(chS[i]) - 1);
				if (map.get(chS[i]) == 0)
					count++;
			}
			if (i >= pLen - 1) {
				int leftIndex = i - pLen + 1;
				if (count == map.size())
					return true;
				if (map.containsKey(chS[leftIndex])) {
					if (map.get(chS[leftIndex]) == 0)
						count--;
					map.put(chS[leftIndex], map.get(chS[leftIndex]) + 1);
				}
			}
		}
		return false;
	}

	public List<Integer> findAnagrams(String s, String p) {
		List<Integer> res = new ArrayList<>();
		Map<Character, Integer> map = new HashMap<>();
		char[] chS = s.toCharArray();
		char[] chP = p.toCharArray();
		int pLen = chP.length;
		for (Character c : chP) {
			map.put(c, map.getOrDefault(c, 0) + 1);
		}
		int count = 0;
		for (int i = 0; i < chS.length; i++) {
			if (map.containsKey(chS[i])) {
				map.put(chS[i], map.get(chS[i]) - 1);
				if (map.get(chS[i]) == 0)
					count++;
			}
			if (i >= pLen - 1) {
				int leftIndex = i - pLen + 1;
				if (count == map.size())
					res.add(leftIndex);
				if (map.containsKey(chS[leftIndex])) {
					if (map.get(chS[leftIndex]) == 0)
						count--;
					map.put(chS[leftIndex], map.get(chS[leftIndex]) + 1);
				}
			}
		}
		return res;
	}

	public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
		List<TreeNode> res = new ArrayList<>();
		Set<Integer> set = new HashSet<>();
		for (int i : to_delete) {
			set.add(i);
		}
		if (!set.contains(root.val)) {
			res.add(root);
		}
		helper(root, null, true, set, res);
		return res;
	}

	public void helper(TreeNode root, TreeNode pre, boolean dir, Set<Integer> set, List<TreeNode> res) {
		if (root == null)
			return;
		if (set.contains(root.val)) {
			if (root.left != null) {
				if (!set.contains(root.left.val)) {
					res.add(root.left);
				}
				helper(root.left, null, true, set, res);
			}
			if (root.right != null) {
				if (!set.contains(root.right.val)) {
					res.add(root.right);
				}
				helper(root.right, null, false, set, res);
			}
			if (pre != null) {
				if (dir) {
					pre.left = null;
				} else {
					pre.right = null;
				}
			}
		} else {
			if (root.left != null)
				helper(root.left, root, true, set, res);
			if (root.right != null)
				helper(root.right, root, false, set, res);
		}
	}

	public static String[] bfs(int[][] matrix, int x, int y, int[][] target, int max) {
		return new String[0];
	}

	public static int minKnightMoves(int x, int y) {
		Map<String, Integer> map = new HashMap<>();
		map.put("0,0", 0);
		map.put("1,0", 3);
		map.put("1,1", 2);
		map.put("2,0", 2);
		return helper2(x, y, map);
	}

	private static int helper2(int x, int y, Map<String, Integer> map) {
		x = Math.abs(x);
		y = Math.abs(y);
		if (x < y) {
			int temp = x;
			x = y;
			y = temp;
		}
		String s = x + "," + y;
		if (map.containsKey(s))
			return map.get(s);
		int temp = Math.min(helper2(x - 2, y - 1, map), helper2(x - 1, y - 2, map)) + 1;
		map.put(s, temp);
		return temp;
	}

	public boolean isAnagram(String s, String t) {
		if (s.length() != t.length())
			return false;
		char[] ch1 = s.toCharArray();
		char[] ch2 = t.toCharArray();
		int len = ch1.length;
		int[] count = new int[26];
		for (int i = 0; i < len; i++) {
			count[ch1[i] - 'a']++;
			count[ch2[i] - 'a']--;
		}
		for (int i = 0; i < len; i++) {
			if (count[i] != 0)
				return false;
		}
		return true;
	}

	public int shortestDistance(String[] words, String word1, String word2) {
		boolean flag1 = true;
		int len = words.length;
		int min = Integer.MAX_VALUE;
		int index1 = 0, index2 = 0;
		for (int i = 0; i < len; i++) {
			if (words[i].equals(word1)) {
				flag1 = true;
				index1 = i;
				break;
			} else if (words[i].equals(word2)) {
				flag1 = false;
				index1 = i;
				index2 = i;
				break;
			}
		}
		for (int i = index1 + 1; i < len; i++) {
			if (words[i].equals(word1)) {
				if (flag1) {

				} else {
					min = Math.min(min, i - index2);
				}
				flag1 = true;
				index1 = i;
			} else if (words[i].equals(word2)) {
				if (flag1) {
					min = Math.min(min, i - index1);
				} else {

				}
				flag1 = false;
				index2 = i;
			}
		}
		return min;
	}

	public int maximalSquare(char[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			return 0;
		int row = matrix.length, col = matrix[0].length;
		int[][] dp = new int[row][col];
		int size = 0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (i == 0 || j == 0) {
					dp[i][j] = matrix[i][j] - '0';
				} else if (matrix[i][j] == '0') {
					dp[i][j] = 0;
				} else {
					dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
				}
				if (size < dp[i][j])
					size = dp[i][j];
			}
		}
		return size * size;
	}

	public void merge(int[] nums1, int m, int[] nums2, int n) {
		int prt1 = m - 1, prt2 = n - 1, prt3 = m + n - 1;
		while (prt1 >= 0 && prt2 >= 0) {
			if (nums1[prt1] > nums2[prt2]) {
				nums1[prt3] = nums1[prt1--];
			} else {
				nums1[prt3] = nums2[prt2--];
			}
			prt3--;
		}
		if (prt1 == -1) {
			while (prt2 >= 0) {
				nums1[prt3--] = nums2[prt2--];
			}
		} else {
			while (prt1 >= 0) {
				nums1[prt3--] = nums1[prt1--];
			}
		}
	}

	public int lengthOfLastWord(String s) {
		String[] ss = s.split(" ");
		if (ss.length == 0)
			return 0;
		return ss[ss.length - 1].length();
	}

	public int minSwaps(int[] data) {
		int len = data.length;
		if (len < 3)
			return 0;
		int ones = 0;
		for (int i : data) {
			ones += data[i];
		}
		if (ones == 0)
			return 0;
		int l = 0, r = 0, max = 0, count = 0;
		while (r < len) {
			while (r < len && r - l < ones) {
				count += data[r];
				r++;
			}
			max = Math.max(max, count);
			if (data[l++] == 1)
				count--;
		}
		return ones - max;
	}

	public static double random10() {
		double[] result = { 0.0 };
		helper(10, 0, result);
		return result[0];
	}

	private static void helper(int curSum, int curNum, double[] result) {
		curNum++;
		for (int i = 1; i <= 10; i++) {
			int temp = curSum - i;
			if (temp <= 0) {
				result[0] += (10 - i + 1) * pow2(0.1, curNum) * curNum;
				return;
			} else {
				helper(temp, curNum, result);
			}
		}
	}

	private static double pow2(double x, int n) {
		if (n == 1)
			return x;
		double y = pow2(x, n / 2);
		return n % 2 == 0 ? y * y : y * y * x;
	}

	public static int level(int curSum) {
		if (curSum <= 0)
			return 0;
		int val = random10Second();
		return level(curSum - val) + 1;
	}

	public static int random10Second() {
		int val = (int) (Math.random() * 10 + 1);
		return val;
	}

	public int[] exclusiveTime(int n, List<String> logs) {
		Stack<Integer> stack = new Stack<>();
		int[] res = new int[n];
		String[] s = logs.get(0).split(":");
		stack.push(Integer.valueOf(s[0]));
		int prev = Integer.valueOf(s[2]);
		int temp;
		for (int i = 1; i < logs.size(); i++) {
			s = logs.get(i).split(":");
			if (s[1].equals("begin")) {
				temp = Integer.valueOf(s[2]);
				if (!stack.isEmpty())
					res[stack.peek()] += temp - prev;
				prev = temp;
				stack.push(Integer.valueOf(s[0]));
			} else {
				temp = Integer.valueOf(s[2]);
				res[stack.pop()] += temp - prev + 1;
				prev = temp + 1;
			}
		}
		return res;
	}

//	44 best solution
	public boolean isMatch3(String s, String p) {
		int i = 0;
		int j = 0;
		int starIndex = -1;
		int iIndex = -1;

		while (i < s.length()) {
			if (j < p.length() && (p.charAt(j) == '?' || p.charAt(j) == s.charAt(i))) {
				++i;
				++j;
			} else if (j < p.length() && p.charAt(j) == '*') {
				starIndex = j;
				iIndex = i;
				j++;
			} else if (starIndex != -1) {
				j = starIndex + 1;
				i = iIndex + 1;
				iIndex++;
			} else {
				return false;
			}
		}

		while (j < p.length() && p.charAt(j) == '*') {
			++j;
		}

		return j == p.length();
	}

	public static boolean isMatch2(String s, String p) {
		Set<String> set = new HashSet<>();
		return helper(s, p, 0, 0, set);
	}

	private static boolean helper(String s, String p, int i, int j, Set<String> set) {
		if (set.contains(i + "," + j))
			return false;
		int sLen = s.length(), pLen = p.length();
		if (i == sLen && j == pLen)
			return true;
		if (j == pLen) {
			set.add(i + "," + j);
			return false;
		}
		if (i == sLen) {
			int m = j;
			while (m < pLen && p.charAt(m) == '*') {
				m++;
			}
			if (m == pLen)
				return true;
			set.add(i + "," + j);
			return false;
		}
		if (p.charAt(j) == '?' || p.charAt(j) == s.charAt(i)) {
			if (helper(s, p, i + 1, j + 1, set)) {
				return true;
			} else {
				set.add(i + "," + j);
				return false;
			}
		}
		if (p.charAt(j) == '*') {
			for (int m = i; m <= sLen; m++) {
				if (helper(s, p, m, j + 1, set)) {
					return true;
				}
			}
		}
		set.add(i + "," + j);
		return false;
	}

	public List<List<Integer>> verticalOrder(TreeNode root) {
		List<List<Integer>> list = new ArrayList<>();
		if (root == null)
			return list;
		Queue<TreeNode> queue = new LinkedList<>();
		Map<TreeNode, Integer> map = new HashMap<>();
		Map<Integer, List<Integer>> res = new TreeMap<>();
		int index = 0;
		map.put(root, index);
		queue.offer(root);
		while (!queue.isEmpty()) {
			TreeNode cur = queue.poll();
			int temp = map.get(cur);
			if (res.containsKey(temp)) {
				res.get(temp).add(cur.val);
			} else {
				List<Integer> now = new ArrayList<>();
				now.add(cur.val);
				res.put(temp, now);
			}
			if (cur.left != null) {
				map.put(cur.left, temp - 1);
				queue.offer(cur.left);
			}
			if (cur.right != null) {
				map.put(cur.right, temp + 1);
				queue.offer(cur.right);
			}
		}
		return new ArrayList<>(res.values());
	}

//	264
	public int nthUglyNumber(int n) {
		if (n < 0) {
			return -1;
		}
		int[] ugly = new int[n];
		ugly[0] = 1;

		int i2 = 0;
		int i3 = 0;
		int i5 = 0;

		for (int i = 1; i < n; i++) {
			ugly[i] = Math.min(ugly[i2] * 2, Math.min(ugly[i3] * 3, ugly[i5] * 5));
			if (ugly[i] == ugly[i2] * 2) {
				i2++;
			}
			if (ugly[i] == ugly[i3] * 3) {
				i3++;
			}
			if (ugly[i] == ugly[i5] * 5) {
				i5++;
			}
		}
		return ugly[n - 1];
	}

	public int longestPalindromeSubseq2(String s) {
		int len = s.length();
		int[][] dp = new int[len][len];
		for (int i = 0; i < len; i++) {
			dp[i][i] = 1;
		}
		for (int i = 0; i < len - 1; i++) {
			if (s.charAt(i) == s.charAt(i + 1)) {
				dp[i][i + 1] = 2;
			} else {
				dp[i][i + 1] = 1;
			}
		}
		for (int dis = 3; dis <= len; dis++) {
			int maxIndex = len - dis + 1;
			for (int i = 0; i < maxIndex; i++) {
				int right = i + dis - 1;
				if (s.charAt(i) == s.charAt(right)) {
					dp[i][right] = 2 + dp[i + 1][right - 1];
				} else {
					dp[i][right] = Math.max(dp[i + 1][right], dp[i][right - 1]);
				}
			}
		}
		return dp[0][len - 1];
	}

	public int longestPalindromeSubseq(String s) {
		return dfs(s, 0, s.length() - 1, new int[s.length()][s.length()]);
	}

	private int dfs(String s, int left, int right, int[][] visited) {
		if (left > right)
			return 0;
		if (left == right)
			return 1;
		if (visited[left][right] != 0) {
			return visited[left][right];
		}
		int max;
		if (s.charAt(left) == s.charAt(right)) {
			max = 2 + dfs(s, left + 1, right - 1, visited);
		} else {
			max = Math.max(dfs(s, left + 1, right, visited), dfs(s, left, right - 1, visited));
		}
		visited[left][right] = max;
		return max;
	}

	public int longestCommonSubsequence(String text1, String text2) {
		if (text1 == null || text1.length() == 0 || text2 == null || text2.length() == 0)
			return 0;
		int len1 = text1.length(), len2 = text2.length();
		int[][] dp = new int[len1 + 1][len2 + 1];
		for (int i = 1; i <= len1; i++) {
			for (int j = 1; j <= len2; j++) {
				if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
					dp[i][j] = dp[i - 1][j - 1] + 1;
				} else {
					dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
		return dp[len1][len2];
	}

//	leetcode 375
	public int getMoneyAmount(int n) {
		int[][] dp = new int[n + 1][n + 1];

		for (int len = 2; len <= n; len++) {
			for (int start = 1; start <= n - len + 1; start++) {
				int minres = Integer.MAX_VALUE;
				for (int piv = start + (len - 1) / 2; piv < start + len - 1; piv++) {
					int res = piv + Math.max(dp[start][piv - 1], dp[piv + 1][start + len - 1]);
					minres = Math.min(res, minres);
				}
				dp[start][start + len - 1] = minres;
			}

		}
		return dp[1][n];
	}

	public static int[] numSmallerByFrequency(String[] queries, String[] words) {
		int len = words.length;
		int[] count = new int[len];
		for (int i = 0; i < len; i++) {
			count[i] = f(words[i]);
		}
		int lenQ = queries.length;
		int[] res = new int[lenQ];
		Arrays.sort(count);
		for (int i = 0; i < lenQ; i++) {
			int temp = f(queries[i]);
			int index = binarySearch(count, temp);
			res[i] = len - index;
		}
		return res;
	}

	private static int binarySearch(int[] count, int target) {
		int left = 0;
		int right = count.length - 1;
		while (left < right - 1) {
			int mid = left + (right - left) / 2;
			if (count[mid] <= target) {
				left = mid;
			} else {
				right = mid;
			}
		}
		if (count[right] <= target)
			return right + 1;
		if (count[left] > target)
			return left;
		return right;
	}

	private static int f(String s) {
		char[] ch = s.toCharArray();
		int len = ch.length;
		char c = 'z';
		int count = 0;
		for (int i = 0; i < len; i++) {
			if (ch[i] == c) {
				count++;
			} else if (ch[i] < c) {
				c = ch[i];
				count = 1;
			}
		}
		return count;
	}

	public TreeNode buildTree2(int[] inorder, int[] postorder) {
		Map<Integer, Integer> map = new HashMap<>();
		int len = inorder.length;
		for (int i = 0; i < len; i++) {
			map.put(inorder[i], i);
		}
		return helper2(inorder, postorder, 0, len - 1, new int[] { len - 1 }, map);
	}

	private TreeNode helper2(int[] inorder, int[] postorder, int left, int right, int[] index,
			Map<Integer, Integer> map) {
		if (left > right)
			return null;
		int root = postorder[index[0]--];
		TreeNode node = new TreeNode(root);
		int root_index = map.get(root);
		node.right = helper(inorder, postorder, root_index + 1, right, index, map);
		node.left = helper(inorder, postorder, left, root_index - 1, index, map);
		return node;
	}

	public TreeNode buildTree(int[] preorder, int[] inorder) {
		Map<Integer, Integer> map = new HashMap<>();
		int len = inorder.length;
		for (int i = 0; i < len; i++) {
			map.put(inorder[i], i);
		}
		return helper(preorder, inorder, 0, len - 1, new int[] { 0 }, map);
	}

	private TreeNode helper(int[] preorder, int[] inorder, int left, int right, int[] index,
			Map<Integer, Integer> map) {
		if (left > right)
			return null;
		int root = preorder[index[0]++];
		TreeNode node = new TreeNode(root);
		int root_index = map.get(root);
		node.left = helper(preorder, inorder, left, root_index - 1, index, map);
		node.right = helper(preorder, inorder, root_index + 1, right, index, map);
		return node;
	}

	public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
		if (t1 == null && t2 == null)
			return null;
		if (t1 == null)
			return t2;
		if (t2 == null)
			return t1;
		TreeNode node = new TreeNode(t1.val + t2.val);
		node.left = mergeTrees(t1.left, t2.left);
		node.right = mergeTrees(t1.right, t2.right);
		return node;
	}

	public List<TreeNode> generateTrees(int n) {
		if (n == 0)
			return new ArrayList<>();
		Map<List<Integer>, List<TreeNode>> map = new HashMap<>();
		return helper(1, n, map);
	}

	private List<TreeNode> helper(int begin, int end, Map<List<Integer>, List<TreeNode>> map) {
		List<Integer> index = new ArrayList<>();
		index.add(begin);
		index.add(end);
		if (map.containsKey(index))
			return map.get(index);

		List<TreeNode> all = new LinkedList<>();
		if (begin > end) {
			all.add(null);
			return all;
		}
		for (int i = begin; i <= end; i++) {
			List<TreeNode> left = helper(begin, i - 1, map);
			List<TreeNode> right = helper(i + 1, end, map);

			for (TreeNode l : left) {
				for (TreeNode r : right) {
					TreeNode cur = new TreeNode(i);
					cur.left = l;
					cur.right = r;
					all.add(cur);
				}
			}
		}
		map.put(index, all);
		return all;
	}

	public ListNode deleteDuplicates2(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode prepre = dummy;
		ListNode pre = head;
		ListNode cur = head.next;
		while (cur != null) {
			if (cur.val == pre.val) {
				while (cur != null && cur.val == pre.val) {
					cur = cur.next;
				}
				pre = cur;
				prepre.next = pre;
				if (cur == null) {
					break;
				}
				cur = cur.next;
			} else {
				prepre = pre;
				pre = cur;
				cur = cur.next;
			}
		}
		return dummy.next;
	}

	public ListNode deleteDuplicates(ListNode head) {
		if (head == null)
			return null;
		ListNode pre = head;
		ListNode cur = head.next;
		while (cur != null) {
			if (cur.val == pre.val) {
				pre.next = cur.next;
				cur.next = null;
				cur = pre.next;
			} else {
				cur = cur.next;
				pre = pre.next;
			}
		}
		return head;
	}

	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
		if (obstacleGrid == null || obstacleGrid.length == 0 || obstacleGrid[0] == null || obstacleGrid[0].length == 0)
			return 0;
		int row = obstacleGrid.length, col = obstacleGrid[0].length;
		if (obstacleGrid[0][0] == 1 || obstacleGrid[row - 1][col - 1] == 1)
			return 0;
		int[][] res = new int[row][col];
		res[row - 1][col - 1] = 1;
		for (int i = row - 2; i >= 0; i--) {
			if (obstacleGrid[i][col - 1] != 1) {
				res[i][col - 1] = res[i + 1][col - 1];
			} else {
				res[i][col - 1] = 0;
			}

		}
		for (int j = col - 2; j >= 0; j--) {
			if (obstacleGrid[row - 1][j] != 1) {
				res[row - 1][j] = res[row - 1][j + 1];
			} else {
				res[row - 1][j] = 0;
			}

		}
		res[row - 1][col - 1] = 1;
		for (int i = row - 2; i >= 0; i--) {
			for (int j = col - 2; j >= 0; j--) {
				if (obstacleGrid[i][j] != 1) {
					res[i][j] = res[i + 1][j] + res[i][j + 1];
				} else {
					res[i][j] = 0;
				}
			}
		}
		return res[0][0];
	}

	public int removeDuplicates(int[] nums) {
		int len = nums.length;
		if (len <= 2)
			return len;
		int index = 2;
		for (int i = 2; i < len; i++) {
			if (nums[i] != nums[index - 2]) {
				nums[index++] = nums[i];
			}
		}
		return index;
	}

	public void sortColors(int[] nums) {
		if (nums == null || nums.length == 0)
			return;
		int len = nums.length;
		int p0 = 0, p2 = len - 1, cur = 0;
		while (cur <= p2) {
			if (nums[cur] == 0) {
				swap(nums, p0, cur);
				p0++;
				cur++;
			} else if (nums[cur] == 2) {
				swap(nums, p2, cur);
				p2--;
			} else {
				cur++;
			}
		}
	}

	private void swap(int[] nums, int left, int right) {
		int temp = nums[left];
		nums[left] = nums[right];
		nums[right] = temp;
	}

	public boolean canJump(int[] nums) {
		if (nums == null || nums.length <= 1)
			return true;
		int maxIndex = nums[0];
		int target = nums.length - 1;
		int i = 1;
		while (i <= maxIndex) {
			int index = nums[i] + i;
			if (index > maxIndex)
				maxIndex = index;
			if (maxIndex >= target)
				return true;
			i++;
		}
		return false;
	}

	public static int mctFromLeafValues(int[] arr) {
		int[] max = new int[1];
		helper(arr, 0, arr.length - 1, max);
		return max[0];
	}

	private static int helper(int[] arr, int left, int right, int[] max) {
		if (left > right)
			return 0;
		if (left == right)
			return arr[left];
		if (right - left == 1) {
			max[0] += arr[left] * arr[right];
			return arr[left] > arr[right] ? arr[left] : arr[right];
		}
		int maxIndex = left;
		for (int i = left + 1; i <= right; i++) {
			if (arr[i] > arr[maxIndex]) {
				maxIndex = i;
			}
		}
		int l = helper(arr, left, maxIndex - 1, max);
		int r = helper(arr, maxIndex + 1, right, max);
		if (l != 0 || r != 0) {
			max[0] += (l + r) * arr[maxIndex];
		}
		return arr[maxIndex];
	}

	public int maxLevelSum(TreeNode root) {
		Queue<TreeNode> queue = new LinkedList<>();
		queue.offer(root);
		int max = 0;
		int res = 0;
		int level = 1;
		while (!queue.isEmpty()) {
			int size = queue.size();
			int sum = 0;
			while (size-- > 0) {
				TreeNode cur = queue.poll();
				sum += cur.val;
				if (cur.left != null)
					queue.offer(cur.left);
				if (cur.right != null)
					queue.offer(cur.right);
			}
			if (max < sum) {
				res = level;
				max = sum;
			}
			level++;
		}
		return res;
	}

	public static int calculateTime(String keyboard, String word) {
		char[] ch = keyboard.toCharArray();
		int[] realPos = new int[26];
		for (int i = 0; i < 26; i++) {
			realPos[ch[i] - 'a'] = i;
		}
		char[] wordCh = word.toCharArray();
		int count = 0;
		int pre = 0;
		int realIndex = 0;
		for (char c : wordCh) {
			realIndex = realPos[c - 'a'];
			count += Math.abs(realIndex - pre);
			pre = realIndex;
		}
		return count;
	}

	public static boolean reachingPoints(int sx, int sy, int tx, int ty) {
		if (sx == tx && sy == ty)
			return true;
		if (tx == ty)
			return false;
		if (tx < sx || ty < sy)
			return false;
		if (tx == sx)
			return (ty - sy) % tx == 0;
		if (ty == sy)
			return (tx - sx) % ty == 0;
		return tx < ty ? reachingPoints(sx, sy, tx, ty % tx) : reachingPoints(sx, sy, tx % ty, ty);
	}

	public TreeNode sortedListToBST(ListNode head) {
		int count = 0;
		ListNode cur = head;
		while (cur != null) {
			cur = cur.next;
			count++;
		}
		ListNode[] list = { head };
		return helper(list, 0, count - 1);
	}

	private TreeNode helper(ListNode[] head, int l, int r) {
		if (l > r)
			return null;
		int mid = l + (r - l) / 2;
		TreeNode left = helper(head, l, mid - 1);
		TreeNode node = new TreeNode(head[0].val);
		node.left = left;
		head[0] = head[0].next;
		node.right = helper(head, mid + 1, r);
		return node;
	}

	public ListNode plusOne(ListNode head) {
		int out = addOne(head);
		if (out == 1) {
			ListNode newHead = new ListNode(1);
			newHead.next = head;
			head = newHead;
		}
		return head;
	}

	private int addOne(ListNode head) {
		if (head == null)
			return 1;
		int next = addOne(head.next);
		if (next + head.val >= 10) {
			head.val = 0;
			return 1;
		} else {
			head.val = head.val + next;
			return 0;
		}
	}

	public int[] plusOne(int[] digits) {
		int len = digits.length;
		for (int i = len - 1; i >= 0; i--) {
			digits[i] += 1;
			if (digits[i] == 10) {
				digits[i] = 0;
			} else {
				break;
			}
		}
		if (digits[0] == 0) {
			int[] res = new int[len + 1];
			res[0] = 1;
			return res;
		}
		return digits;
	}

	public static boolean canIWin(int maxChoosableInteger, int desiredTotal) {
		if ((1 + maxChoosableInteger) * maxChoosableInteger / 2 < desiredTotal)
			return false;
		int map = (1 << maxChoosableInteger) - 1;
		Boolean[] mem = new Boolean[map + 1];
		return canIWinHelper(maxChoosableInteger, desiredTotal, map, mem);
	}

	private static boolean canIWinHelper(int maxCanChoose, int dt, int map, Boolean[] mem) {
		if (mem[map] != null)
			return mem[map];
		for (int i = 1; i <= maxCanChoose; i++) {
			int mask = 1 << (i - 1);
			if ((map & mask) != 0) {
				if (dt - i <= 0) {
					mem[map] = true;
					return true;
				}
				int newMap = map - mask;
				if (!canIWinHelper(maxCanChoose, dt - i, newMap, mem)) {
					mem[map] = true;
					return true;
				}
			}
		}
		mem[map] = false;
		return false;
	}

	public boolean canCross(int[] stones) {
		if (stones == null || stones.length == 0)
			return false;
		if (stones.length == 1)
			return true;
		if (stones[1] != 1)
			return false;
		int len = stones.length;
		Map<Integer, Boolean>[] map = new HashMap[stones.length];
		for (int i = 0; i < len; i++) {
			map[i] = new HashMap<Integer, Boolean>();
		}
		return canCrossHelper(stones, 1, 1, map);
	}

	private boolean canCrossHelper(int[] stones, int index, int k, Map<Integer, Boolean>[] map) {
		if (index == stones.length - 1) {
			return true;
		}
		Map<Integer, Boolean> cur = map[index];
		if (cur.containsKey(k))
			return cur.get(k);
		int len = stones.length;
		for (int i = index + 1; i < len; i++) {
			int distance = stones[i] - stones[index];
			if (distance < k - 1)
				continue;
			if (distance > k + 1) {
				cur.put(k, false);
				return false;
			}
			if (canCrossHelper(stones, i, distance, map)) {
				cur.put(k, true);
				return true;
			}
		}
		cur.put(k, false);
		return false;
	}

	public static String removeKdigits(String num, int k) {
		if (num == null || k >= num.length())
			return "0";
		int pos = 0;
		char[] res = new char[num.length()];
		int len = num.length() - k; //////
		for (char c : num.toCharArray()) {
			while (pos > 0 && k > 0 && res[pos - 1] > c) {
				pos--;
				k--;
			}
			res[pos++] = c;
		}
		int start = 0;
		while (start < len && res[start] == '0')
			start++;
		return start == pos ? "0" : new String(res, start, len - start);
	}

	public TreeNode sortedArrayToBST(int[] nums) {
		return toBST(nums, 0, nums.length - 1);
	}

	private TreeNode toBST(int[] nums, int left, int right) {
		if (left > right)
			return null;
		int mid = left + (right - left) / 2;
		TreeNode root = new TreeNode(nums[mid]);
		root.left = toBST(nums, left, mid - 1);
		root.right = toBST(nums, mid + 1, right);
		return root;
	}

	public static boolean canWin(String s) {
		return canWinHelper(s.toCharArray(), new HashMap<String, Boolean>());
	}

	private static boolean canWinHelper(char[] ch, Map<String, Boolean> mem) {
		String str = new String(ch);
		if (mem.containsKey(str))
			return mem.get(str);
		int len = ch.length;
		for (int i = 0; i < len - 1; i++) {
			if (ch[i] == '+' && ch[i + 1] == '+') {
				ch[i] = '-';
				ch[i + 1] = '-';
				boolean res = canWinHelper(ch, mem);
				ch[i] = '+';
				ch[i + 1] = '+';
				if (!res) {
					mem.put(str, true);
					return true;
				}
			}
		}
		mem.put(str, false);
		return false;
	}

	public static int water(int[] arr, int C) {
		int[] count = { 1, Integer.MAX_VALUE };
		helper(arr, C, C, 0, count);
		return count[1];
	}

	private static void helper(int[] arr, int C, int curC, int index, int[] count) {
		int len = arr.length;
		if (curC < 0)
			return;
		if (index == len - 1) {
			if (arr[index] <= curC) {
				count[1] = Math.min(count[0], count[1]);
			} else {
				count[1] = Math.min(count[0] + 2 * (index + 1), count[1]);
			}
			return;
		}

		int curCount = count[0];
		if (curC == 0 || arr[index] > curC) {
			count[0] += 2 * (index + 1);
			helper(arr, C, C, index, count);
			count[0] = curCount;
		} else {
			count[0]++;
			helper(arr, C, curC - arr[index], index + 1, count);
			count[0] += 2 * (index + 1);
			helper(arr, C, C, index + 1, count);
			count[0] = curCount;
		}
	}

	public List<String> wordBreak3(String s, List<String> wordDict) {
		List<String> res = new ArrayList<>();
		Set<String> set = new HashSet<>();
		for (String str : wordDict) {
			set.add(str);
		}
		boolean[] mem = new boolean[s.length()];
		helper(s, set, res, new StringBuilder(), 0, mem);
		return res;
	}

	private void helper(String word, Set<String> dict, List<String> res, StringBuilder path, int idx, boolean[] mem) {
		int len = word.length();
		if (idx == len) {
			path.setLength(path.length() - 1);
			res.add(path.toString());
			return;
		}
		if (mem[idx]) {
			return;
		}
		int size = res.size();
		for (int i = idx + 1; i <= len; i++) {
			String str = word.substring(idx, i);
			if (dict.contains(str)) {
				int pathLen = path.length();
				path.append(str + " ");
				helper(word, dict, res, path, i, mem);
				path.setLength(pathLen);
			}
		}
		if (res.size() == size) {
			mem[idx] = true;
		}
	}

	public List<List<Integer>> subsets3(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		subsets3helper(nums, res, new ArrayList<Integer>(), 0);
		return res;
	}

	private void subsets3helper(int[] nums, List<List<Integer>> res, List<Integer> sofar, int index) {
		if (index == nums.length) {
			res.add(new ArrayList<Integer>(sofar));
			return;
		}
//		not add index's value
		subsets3helper(nums, res, sofar, index + 1);
//		add index's value
		sofar.add(nums[index]);
		subsets3helper(nums, res, sofar, index + 1);
		sofar.remove(sofar.size() - 1);
	}

	public String serialize(TreeNode root) {
		if (root == null)
			return "[null]";
		Queue<TreeNode> queue = new LinkedList<>();
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		queue.offer(root);
		while (!queue.isEmpty()) {
			int size = queue.size();
			while (size-- > 0) {
				TreeNode cur = queue.poll();
				if (cur == null) {
					sb.append("null,");
				} else {
					sb.append(cur.val + ",");
					queue.offer(cur.left);
					queue.offer(cur.right);
				}
			}
		}
		sb.setLength(sb.length() - 1);
		sb.append("]");

		return sb.toString();
	}

	// Decodes your encoded data to tree.
	public TreeNode deserialize(String data) {
		if (data.equals("[null]"))
			return null;
		data = data.substring(1, data.length() - 1);
		String[] arr = data.split(",");
		int i = 0;
		TreeNode root = new TreeNode(Integer.parseInt(arr[i++].trim()));
		Queue<TreeNode> q = new LinkedList<>();
		q.add(root);
		while (!q.isEmpty()) {
			int s = q.size();
			for (int j = 0; j < s; j++) {
				TreeNode temp = q.poll();
				String s2 = arr[i++].trim();
				String s3 = arr[i++].trim();
				if (!s2.equals("null")) {
					temp.left = new TreeNode(Integer.parseInt(s2));
					q.add(temp.left);
				}
				if (!s3.equals("null")) {
					temp.right = new TreeNode(Integer.parseInt(s3));
					q.add(temp.right);
				}
			}
		}
		return root;
	}

	public int minDiffInBST(TreeNode root) {
		int[] min = { Integer.MAX_VALUE };
		TreeNode[] pre = { null };
		inorder(root, min, pre);
		return min[0];
	}

	private void inorder(TreeNode root, int[] min, TreeNode[] pre) {
		if (root == null)
			return;
		inorder(root.left, min, pre);

		if (pre[0] != null) {
			min[0] = Math.min(min[0], root.val - pre[0].val);
		}
		pre[0] = root;
		inorder(root.right, min, pre);
	}

	public String longestPalindrome2(String s) {
		int len = s.length();
		if (len <= 1)
			return s;
		int left = 0, right = 0;
		for (int i = 0; i < len; i++) {
			int l = i, r = i;
			while (r < len - 1 && s.charAt(r + 1) == s.charAt(r)) {
				r++;
				i++;
			}
			while (l > 0 && r < len - 1 && s.charAt(l - 1) == s.charAt(r + 1)) {
				l--;
				r++;
			}
			if (r - l > right - left) {
				right = r;
				left = l;
			}
		}
		return s.substring(left, right + 1);
	}

	public boolean canPermutePalindrome(String s) {
		char[] ch = s.toCharArray();
		int[] count = new int[256];
		for (char c : ch) {
			count[c]++;
		}
		int odd = 0;
		for (int i = 0; i < 256; i++) {
			if (count[i] > 0) {
				if (count[i] % 2 != 0) {
					odd++;
				}
			}
		}
		return odd <= 1;
	}

	public int longestPalindrome(String s) {
		char[] ch = s.toCharArray();
		int[] count = new int[52];
		for (char c : ch) {
			if (c - 'a' >= 0 && c - 'z' <= 0) {
				count[c - 'a']++;
			} else if (c - 'A' >= 0 && c - 'Z' <= 0) {
				count[c - 'A' + 26]++;
			}
		}
		boolean flag = false;
		int res = 0;
		for (int i = 0; i < 52; i++) {
			if (count[i] > 0) {
				if (count[i] % 2 == 0) {
					res += count[i];
				} else {
					res += count[i] - 1;
					flag = true;
				}
			}
		}
		if (flag)
			res += 1;
		return res;
	}

	public int maxPoints(int[][] points) {
		if (points == null || points.length == 0)
			return 0;
		if (points.length == 1)
			return 1;
		int max = 0;
		int len = points.length;
		for (int i = 0; i < len; i++) {
			int same = 1;
			int horizon = 0;
			int[] fixP = points[i];
			int curMax = 0;
			Map<Double, Integer> map = new HashMap<>();
			for (int j = i + 1; j < len; j++) {
				int[] curP = points[j];
				if (curP[0] == fixP[0] && curP[1] == fixP[1]) {
					same++;
				} else if (curP[1] == fixP[1]) {
					horizon++;
				} else {
					double slope = (double) (curP[0] - fixP[0]) / (curP[1] - fixP[1]) + 0.0;
					map.put(slope, map.getOrDefault(slope, 0) + 1);
					curMax = Math.max(curMax, map.get(slope));
				}
			}
			max = Math.max(max, Math.max(curMax, horizon) + same);
		}
		return max;
	}

	public static int maxEnvelopes(int[][] envelopes) {
		if (envelopes == null || envelopes.length == 0 || envelopes[0] == null || envelopes[0].length == 0)
			return 0;
		Arrays.sort(envelopes, new Comparator<int[]>() {
			@Override
			public int compare(int[] o1, int[] o2) {
				if (o1[0] == o2[0]) {
					return o2[1] - o1[1];
				}
				return o1[0] - o2[0];
			}
		});

		List<Integer> list = new ArrayList<>();
		int len = envelopes.length;
		for (int i = 0; i < len; i++) {
			int index = binarySearch2(list, envelopes[i][1]);
			if (index < list.size()) {
				list.set(index, envelopes[i][1]);
			} else {
				list.add(envelopes[i][1]);
			}
		}
		return list.size();
	}

	private static int binarySearch2(List<Integer> list, int target) {
		if (list.size() == 0)
			return 1;
		int left = 0, right = list.size() - 1;
		while (left < right - 1) {
			int mid = left + (right - left) / 2;
			int temp = list.get(mid);
			if (list.get(mid) >= target) {
				right = mid;
			} else {
				left = mid;
			}
		}
		if (list.get(right) < target) {
			return right + 1;
		}
		if (list.get(left) < target) {
			return right;
		}
		return left;
	}

	public int findMinStep(String board, String hand) {
		if (board == null || board.length() == 0 || hand == null || hand.length() == 0)
			return -1;
		Map<Character, Integer> map = new HashMap<>();
		for (char c : hand.toCharArray()) {
			map.put(c, map.getOrDefault(c, 0) + 1);
		}
		int[] min = { hand.length() + 1 };
		findMinStepHelper(board, map, 0, min);
		return min[0] == hand.length() + 1 ? -1 : min[0];
	}

	private void findMinStepHelper(String board, Map<Character, Integer> hand, int count, int[] min) {
		int len = board.length();
		if (len == 0) {
			if (min[0] > count) {
				min[0] = count;
			}
			return;
		}
		for (int i = 0; i < len; i++) {
			char c = board.charAt(i);
			Integer num = hand.get(c);
			if (num == null)
				continue;
			if (i < len - 1 && board.charAt(i + 1) == c) {
				if (num > 1) {
					hand.put(c, num - 1);
				} else {
					hand.remove(c);
				}
				String newBoard = removeBoard(board, i - 1, i + 2);
				findMinStepHelper(newBoard, hand, count + 1, min);
				hand.put(c, num);
			} else if (num >= 2) {
				if (num > 2) {
					hand.put(c, num - 2);
				} else {
					hand.remove(c);
				}
				String newBoard = removeBoard(board, i - 1, i + 1);
				findMinStepHelper(newBoard, hand, count + 2, min);
				hand.put(c, num);
			}
		}
	}

	private String removeBoard(String board, int left, int right) {
		int len = board.length();
		while (left >= 0 && right < len) {
			char c = board.charAt(left);
			int count = 0;
			int l = left;
			while (l >= 0 && board.charAt(l) == c) {
				count++;
				l--;
			}
			int r = right;
			while (r < len && board.charAt(r) == c) {
				count++;
				r++;
			}
			if (count >= 3) {
				left = l;
				right = r;
			} else {
				break;
			}
		}
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i <= left; i++) {
			sb.append(board.charAt(i));
		}
		for (int i = right; i < len; i++) {
			sb.append(board.charAt(i));
		}
		return sb.toString();
	}

	public List<String> generateAbbreviations(String word) {
		List<String> res = new ArrayList<>();
		generateAbbreviationsHelper(word, 0, 0, res, new StringBuilder());
		return res;
	}

	public void generateAbbreviationsHelper(String word, int index, int count, List<String> res, StringBuilder path) {
		if (index == word.length()) {
			if (count > 0) {
				int len = path.length();
				path.append(count);
				res.add(path.toString());
				path.setLength(len);
			} else {
				res.add(path.toString());
			}
			return;
		}
//		regard this char as a number,  and count 1
		generateAbbreviationsHelper(word, index + 1, count + 1, res, path);
//		keep this char
		int len = path.length();
		if (count > 0) {
			path.append(count);
		}
		path.append(word.charAt(index));
		generateAbbreviationsHelper(word, index + 1, 0, res, path);
		path.setLength(len);
	}

	public static List<Integer> majorityElement2(int[] nums) {
		List<Integer> ans = new LinkedList<>();
		if (nums == null || nums.length == 0)
			return ans;
		int a = 0, b = 0, counta = 0, countb = 0;
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] == a)
				counta++;
			else if (nums[i] == b)
				countb++;
			else if (counta == 0) {
				a = nums[i];
				counta = 1;
			} else if (countb == 0) {
				b = nums[i];
				countb = 1;
			} else {
				counta--;
				countb--;
			}
		}
		counta = 0;
		countb = 0;
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] == a)
				counta++;
			else if (nums[i] == b)
				countb++;
		}
		if (counta > nums.length / 3)
			ans.add(a);
		if (countb > nums.length / 3)
			ans.add(b);
		return ans;
	}

	public int majorityElement(int[] nums) {
		Map<Integer, Integer> map = new HashMap<>();
		int len = nums.length;
		int temp = 0;
		for (int i = 0; i < len; i++) {
			if (map.isEmpty()) {
				map.put(nums[i], 1);
				temp = nums[i];
			} else if (temp == nums[i]) {
				map.put(temp, map.get(temp) + 1);
			} else {
				map.put(temp, map.get(temp) - 1);
				if (map.get(temp) == 0)
					map.clear();
			}
		}
		return temp;
	}

	public int numDistinctIslands(int[][] grid) {
		int row = grid.length, col = grid[0].length;
		boolean[][] visited = new boolean[row][col];
		Set<String> set = new HashSet<>();
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (grid[i][j] == 1 && !visited[i][j]) {
					StringBuilder path = new StringBuilder();
					numDistinctIslandsHelper(grid, i, j, visited, path, 's');
					String temp = path.toString();
					if (!set.contains(temp)) {
						set.add(temp);
					}
				}
			}
		}
		return set.size();
	}

	private void numDistinctIslandsHelper(int[][] grid, int i, int j, boolean[][] visited, StringBuilder path,
			char ch) {
		if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != 1 || visited[i][j]) {
			path.append('x');
			return;
		}
		path.append(ch);
		visited[i][j] = true;
		numDistinctIslandsHelper(grid, i - 1, j, visited, path, 'u');
		numDistinctIslandsHelper(grid, i + 1, j, visited, path, 'd');
		numDistinctIslandsHelper(grid, i, j - 1, visited, path, 'l');
		numDistinctIslandsHelper(grid, i, j + 1, visited, path, 'r');
	}

	public int maxAreaOfIsland(int[][] grid) {
		if (grid == null || grid.length == 0 || grid[0].length == 0)
			return 0;
		int row = grid.length;
		int col = grid[0].length;
		boolean[][] visited = new boolean[row][col];
		int max = 0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (grid[i][j] == 1 && !visited[i][j]) {
					int val = maxAreaOfIslandHelper(grid, i, j, visited);
					max = Math.max(max, val);
				}
			}
		}
		return max;
	}

	private int maxAreaOfIslandHelper(int[][] grid, int i, int j, boolean[][] visited) {
		if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != 1 || visited[i][j]) {
			return 0;
		}
		visited[i][j] = true;
		return maxAreaOfIslandHelper(grid, i - 1, j, visited) + maxAreaOfIslandHelper(grid, i + 1, j, visited)
				+ maxAreaOfIslandHelper(grid, i, j - 1, visited) + maxAreaOfIslandHelper(grid, i, j + 1, visited) + 1;
	}

	public static boolean isPalindrome(String s) {
		int len = s.length();
		if (len == 0)
			return true;
		char[] ch = s.toCharArray();
		int left = 0, right = len - 1;
		while (left < right) {
			char temp1 = ch[left];
			char temp2 = ch[right];
			if (temp1 - 48 < 0 || temp1 - 122 > 0 || temp1 - 90 > 0 && temp1 - 97 < 0
					|| temp1 - 57 > 0 && temp1 - 65 < 0) {
				left++;
				continue;
			}
			if (temp2 - 48 < 0 || temp2 - 122 > 0 || temp2 - 90 > 0 && temp2 - 97 < 0
					|| temp2 - 57 > 0 && temp2 - 65 < 0) {
				right--;
				continue;
			}
			if (temp1 - 65 >= 0 && temp1 - 90 <= 0) {
				temp1 = (char) (temp1 + 32);
			}
			if (temp2 - 65 >= 0 && temp2 - 90 <= 0) {
				temp2 = (char) (temp2 + 32);
			}
			if (temp1 == temp2) {
				left++;
				right--;
			} else {
				return false;
			}
		}
		return true;
	}

	public int[] getCharCountBucket(String word, int[] charCount) {
		int[] charCountBucket = new int[26];
		for (char ch : word.toCharArray()) {
			if (charCount[ch - 97] > 0) {
				charCount[ch - 97]--;
				charCountBucket[ch - 97]++;
			}
		}
		return charCountBucket;
	}

	public List<String> commonChars(String[] A) {
		List<String> result = new ArrayList();
		if (A.length == 0) {
			return new ArrayList<String>();
		}
		int[] charCount = new int[26];
		for (char ch : A[0].toCharArray()) {
			charCount[ch - 97]++;
		}
		for (int i = 1; i < A.length; i++) {
			charCount = getCharCountBucket(A[i], charCount);
		}
		for (int i = 0; i < charCount.length; i++) {
			while (charCount[i] > 0) {
				result.add("" + (char) (i + 97));
				charCount[i]--;
			}
		}
		return result;
	}

	public void reverseString(char[] s) {
		if (s.length == 0)
			return;
		int i = 0, j = s.length - 1;
		while (i < j) {
			swap2(s, i++, j--);
		}
	}

	private void swap2(char[] s, int i, int j) {
		char temp = s[i];
		s[i] = s[j];
		s[j] = temp;
	}

	public int countUnivalSubtrees(TreeNode root) {
		return countUnivalSubtreesHelper(root)[0];
	}

	private int[] countUnivalSubtreesHelper(TreeNode root) {
		if (root == null)
			return new int[] { 0, 0 };
		if (root.left == null && root.right == null)
			return new int[] { 1, 0 };
		int[] left = countUnivalSubtreesHelper(root.left);
		int[] right = countUnivalSubtreesHelper(root.right);
		if (left[1] == 1 || right[1] == 1)
			return new int[] { left[0] + right[0], 1 };
		if (root.left == null) {
			if (root.right.val == root.val) {
				return new int[] { right[0] + 1, 0 };
			} else {
				return new int[] { right[0], 1 };
			}
		}
		if (root.right == null) {
			if (root.left.val == root.val) {
				return new int[] { left[0] + 1, 0 };
			} else {
				return new int[] { left[0], 1 };
			}
		}
		if (root.left.val == root.val && root.right.val == root.val) {
			return new int[] { left[0] + right[0] + 1, 0 };
		}
		return new int[] { left[0] + right[0], 1 };
	}

	public int longestUnivaluePath(TreeNode root) {
		int[] max = { 0 };
		longestUnivalueHelper(root, max);
		return max[0];
	}

	private int longestUnivalueHelper(TreeNode root, int[] max) {
		if (root == null)
			return 0;
		int left = longestUnivalueHelper(root.left, max);
		int right = longestUnivalueHelper(root.right, max);
		int curLeft = 0, curRight = 0;
		if (root.left != null && root.left.val == root.val) {
			curLeft += left + 1;
		}
		if (root.right != null && root.right.val == root.val) {
			curRight += right + 1;
		}
		max[0] = Math.max(max[0], curLeft + curRight);
		return Math.max(curLeft, curRight);
	}

	public static int maxPathSum(TreeNode root) {
		int[] max = new int[] { Integer.MIN_VALUE };
		maxPathSum(root, max);
		return max[0];
	}

	private static int maxPathSum(TreeNode root, int[] max) {
		if (root == null)
			return 0;
		int leftMax = maxPathSum(root.left, max);
		int rightMax = maxPathSum(root.right, max);
		int val = root.val;
		int curMax = Math.max(leftMax, rightMax) + val;
		int tempMax = Math.max(curMax, val);
		max[0] = Math.max(max[0], Math.max(tempMax, leftMax + val + rightMax));
		return tempMax;
	}

	public static int rob2(int[] nums) {
		if (nums == null || nums.length == 0)
			return 0;
		if (nums.length == 1)
			return nums[0];
		return Math.max(robHelper(nums, 0, nums.length - 2), robHelper(nums, 1, nums.length - 1));
	}

	private static int robHelper(int[] nums, int i, int j) {
		int len = nums.length;
		if (j - i == 0)
			return nums[i];
		int prepre, pre, cur;
		prepre = nums[i];
		pre = Math.max(nums[i], nums[i + 1]);
		cur = pre;
		for (int l = i + 2; l <= j; l++) {
			cur = Math.max(nums[l] + prepre, pre);
			prepre = pre;
			pre = cur;
		}
		return cur;
	}

	public int rob(int[] nums) {
		int len = nums.length;
		if (nums == null)
			return 0;
		if (len == 1)
			return nums[0];
		int[] dp = new int[len];
		dp[0] = nums[0];
		dp[1] = Math.max(nums[0], nums[1]);
		for (int i = 2; i < len; i++) {
			dp[i] = Math.max(nums[i] + dp[i - 2], dp[i - 1]);
		}
		return dp[len - 1];
	}

	public static List<String> removeInvalidParentheses(String s) {
		List<String> res = new ArrayList<>();
		char[] ch = s.toCharArray();
		Set<String> set = new HashSet<>();
		int len = s.length();
		int count = 0, l = 0, r = 0;
		for (int i = 0; i < len; i++) {
			if (ch[i] == '(') {
				count++;
			} else if (ch[i] == ')') {
				if (count > 0) {
					count--;
				} else {
					r++;
				}
			}
		}
		count = 0;
		for (int i = len - 1; i >= 0; i--) {
			if (ch[i] == ')') {
				count++;
			} else if (ch[i] == '(') {
				if (count > 0) {
					count--;
				} else {
					l++;
				}
			}
		}
		removeInvalidParentheses(s, res, new StringBuilder(), l, r, 0, 0, set);
		return res;
	}

	private static void removeInvalidParentheses(String s, List<String> res, StringBuilder sb, int l, int r, int delta,
			int index, Set<String> set) {
		if (l < 0 || r < 0 || delta < 0) {
			return;
		}
		if (index == s.length()) {
			if (delta == 0) {
				String ss = sb.toString();
				if (!set.contains(ss)) {
					set.add(ss);
					res.add(ss);
				}
			}
			return;
		}
		int len = sb.length();
		if (s.charAt(index) == '(') {
			sb.append('(');
			removeInvalidParentheses(s, res, sb, l, r, delta + 1, index + 1, set);
			sb.setLength(len);
			removeInvalidParentheses(s, res, sb, l - 1, r, delta, index + 1, set);
		} else if (s.charAt(index) == ')') {
			sb.append(')');
			removeInvalidParentheses(s, res, sb, l, r, delta - 1, index + 1, set);
			sb.setLength(len);
			removeInvalidParentheses(s, res, sb, l, r - 1, delta, index + 1, set);
		} else {
			sb.append(s.charAt(index));
			removeInvalidParentheses(s, res, sb, l, r, delta, index + 1, set);
			sb.setLength(len);
		}

	}

	public static List<String> addOperators(String num, int target) {
		List<String> res = new ArrayList<>();
		addOperatorsHelper(num, target, 0, 0, res, new StringBuilder());
		return res;
	}

	private static void addOperatorsHelper(String nums, int target, long sum, long last, List<String> res,
			StringBuilder path) {
		if (nums.length() == 0) {
			if (sum == target) {
				res.add(path.toString());
			}
			return;
		}
		char[] ch = nums.toCharArray();
		int length = nums.length();
		int len = path.length();
		long val = 0;
		for (int i = 0; i < length; i++) {
			val = val * 10 + nums.charAt(i) - '0';
			if (path.length() != 0) {
				addOperatorsHelper(nums.substring(i + 1), target, sum + val, val, res, path.append("+" + val));
				path.setLength(len);
				addOperatorsHelper(nums.substring(i + 1), target, sum - val, -val, res, path.append("-" + val));
				path.setLength(len);
				addOperatorsHelper(nums.substring(i + 1), target, sum - last + last * val, last * val, res,
						path.append("*" + val));
				path.setLength(len);
			} else {
				addOperatorsHelper(nums.substring(i + 1), target, val, val, res, path.append(val));
				path.setLength(len);
			}
			if (val == 0)
				return;
		}
	}

	public List<Integer> diffWaysToCompute(String input) {
		List<Integer> res = new ArrayList<Integer>();
		for (int i = 0; i < input.length(); i++) {
			char ch = input.charAt(i);
			if (ch == '+' || ch == '-' || ch == '*') {
				List<Integer> left = diffWaysToCompute(input.substring(0, i));
				List<Integer> right = diffWaysToCompute(input.substring(i + 1, input.length()));
				for (int l : left) {
					for (int r : right) {
						switch (ch) {
						case '+':
							res.add(l + r);
							break;
						case '-':
							res.add(l - r);
							break;
						case '*':
							res.add(l * r);
							break;
						}
					}
				}
			}
		}
		if (res.size() == 0)
			res.add(Integer.valueOf(input));
		return res;
	}

	public static List<Integer> findDuplicates(int[] nums) {
		List<Integer> list = new ArrayList<>();
		int[] count = new int[nums.length];
		int len = nums.length;
		for (int i = 0; i < len; i++) {
			if (count[nums[i] - 1] == 1) {
				list.add(nums[i]);
			} else {
				count[nums[i] - 1]++;
			}
		}
		return list;
	}

	public List<String> binaryTreePaths(TreeNode root) {
		List<String> res = new ArrayList<>();
		binaryTreePathsHelper(root, res, new StringBuilder());
		return res;
	}

	private void binaryTreePathsHelper(TreeNode root, List<String> res, StringBuilder path) {
		if (root == null)
			return;
		if (root.left == null && root.right == null) {
			int len = path.length();
			path.append(root.val);
			res.add(path.toString());
			path.setLength(len);
		}
		int len = path.length();
		path.append(root.val + "->");
		binaryTreePathsHelper(root.left, res, path);
		binaryTreePathsHelper(root.right, res, path);
		path.setLength(len);
	}

	public static List<String> restoreIpAddresses(String s) {
		List<String> res = new ArrayList<>();
		restoreIpAddressesHelper(res, new StringBuilder(), s, 0, 0);
		return res;
	}

	public static void restoreIpAddressesHelper(List<String> res, StringBuilder path, String s, int index, int num) {
		if (num == 4) {
			if (index == s.length()) {
				path.setLength(path.length() - 1);
				res.add(path.toString());
			}
			return;
		}
		int len = path.length();
		for (int i = 1; i <= 3; i++) {
			if (index + i > s.length())
				return;
			int val = Integer.valueOf(s.substring(index, index + i));
			if (val <= 255) {
				path.append(val + ".");
				restoreIpAddressesHelper(res, path, s, index + i, num + 1);
				path.setLength(len);
			}
			if (val == 0)
				return;
		}
	}

	public List<String> findWords(char[][] board, String[] words) {
		List<String> res = new ArrayList<>();
		for (String s : words) {
			if (exist(board, s)) {
				res.add(s);
			}
		}
		return res;
	}

	public boolean exist(char[][] board, String word) {
		int row = board.length, col = board[0].length;
		boolean[][] visited = new boolean[row][col];
		char[] ch = word.toCharArray();
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (search(board, ch, 0, i, j, visited)) {
					return true;
				}
			}
		}
		return false;
	}

	private boolean search(char[][] board, char[] word, int index, int i, int j, boolean[][] visited) {
		if (index == word.length)
			return true;
		int row = board.length, col = board[0].length;
		if (i < 0 || i >= row || j < 0 || j >= col || board[i][j] != word[index] || visited[i][j])
			return false;
		visited[i][j] = true;
		boolean res = search(board, word, index + 1, i - 1, j, visited)
				|| search(board, word, index + 1, i + 1, j, visited)
				|| search(board, word, index + 1, i, j - 1, visited)
				|| search(board, word, index + 1, i, j + 1, visited);
		visited[i][j] = false;
		return res;
	}

	public List<String> generateParenthesis2(int n) {
		List<String> res = new ArrayList<>();
		generateParenthesisHelper(res, new StringBuilder(), n, 0, 0);
		return res;
	}

	public void generateParenthesisHelper(List<String> res, StringBuilder sb, int n, int l, int r) {
		if (l == n && l == r) {
			res.add(sb.toString());
			return;
		}
		if (l > n || r > l)
			return;
		sb.append("(");
		generateParenthesisHelper(res, sb, n, l + 1, r);
		sb.setLength(sb.length() - 1);
		sb.append(")");
		generateParenthesisHelper(res, sb, n, l, r + 1);
		sb.setLength(sb.length() - 1);
	}

	public List<List<Integer>> levelOrder(Node root) {
		List<List<Integer>> res = new ArrayList<>();
		if (root == null)
			return res;
		Queue<Node> queue = new LinkedList<>();
		queue.offer(root);
		while (!queue.isEmpty()) {
			int size = queue.size();
			List<Integer> list = new LinkedList<>();
			while (size-- > 0) {
				Node cur = queue.poll();
				list.add(cur.val);
				for (Node n : cur.neighbors) {
					if (n != null)
						queue.offer(n);
				}
			}
			res.add(list);
		}
		return res;
	}

//107
	public List<List<Integer>> levelOrderBottom(TreeNode root) {
		List<List<Integer>> res = new ArrayList<>();
		if (root == null)
			return res;
		Queue<TreeNode> queue = new LinkedList<>();
		queue.offer(root);
		while (!queue.isEmpty()) {
			int size = queue.size();
			List<Integer> list = new LinkedList<>();
			while (size-- > 0) {
				TreeNode cur = queue.poll();
				list.add(cur.val);
				if (cur.left != null)
					queue.offer(cur.left);
				if (cur.right != null)
					queue.offer(cur.right);
			}
			res.add(list);
		}
		Collections.reverse(res);
		return res;
	}

//	Leetcode 116
	public Node2 connect(Node2 root) {
		if (root == null)
			return null;
		Queue<Node2> queue = new LinkedList<>();
		queue.offer(root);
		while (!queue.isEmpty()) {
			int size = queue.size();
			Node2 temp = null;
			while (size-- > 0) {
				Node2 cur = queue.poll();
				cur.next = temp;
				temp = cur;
				if (cur.right != null)
					queue.offer(cur.right);
				if (cur.left != null)
					queue.offer(cur.left);
			}
		}
		return root;
	}

	public int pathSum2(TreeNode root, int sum) {
		if (root == null)
			return 0;
		return pathSumFrom(root, sum) + pathSum2(root.left, sum) + pathSum2(root.right, sum);
	}

	public int pathSumFrom(TreeNode root, int sum) {
		if (root == null)
			return 0;
		return (root.val == sum ? 1 : 0) + pathSumFrom(root.left, sum - root.val)
				+ pathSumFrom(root.right, sum - root.val);
	}

	public List<List<Integer>> pathSum(TreeNode root, int sum) {
		List<List<Integer>> res = new ArrayList<>();
		List<Integer> sofar = new ArrayList<>();
		pathSum(root, res, sofar, sum);
		return res;
	}

	public void pathSum(TreeNode root, List<List<Integer>> res, List<Integer> sofar, int sum) {
		if (root == null)
			return;
		if (root.left == null && root.right == null) {
			if (sum == root.val) {
				sofar.add(root.val);
				res.add(new ArrayList<Integer>(sofar));
				sofar.remove(sofar.size() - 1);
			}
			return;
		}
		int nextSum = sum - root.val;
		sofar.add(root.val);
		pathSum(root.left, res, sofar, nextSum);
		pathSum(root.right, res, sofar, nextSum);
		sofar.remove(sofar.size() - 1);
	}

	public boolean hasPathSum(TreeNode root, int sum) {
		if (root == null)
			return false;
		if (root.left == null && root.right == null)
			return root.val == sum;
		int nextSum = sum - root.val;
		return hasPathSum(root.left, nextSum) || hasPathSum(root.right, nextSum);
	}

	public static List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
		List<List<String>> res = new ArrayList<>();
		Set<String> wordSet = new HashSet<>();
		for (String word : wordList) {
			wordSet.add(word);
		}
		Queue<String> queue = new LinkedList<>();
		HashMap<String, List<String>> graph = new HashMap<String, List<String>>();
		queue.offer(beginWord);
		boolean flag = false;
		while (!queue.isEmpty()) {
			int size = queue.size();
			Set<String> visitedThisLev = new HashSet<>();
			while (size-- > 0) {
				String cur = queue.poll();
				char[] cc = cur.toCharArray();
				for (int i = 0; i < cc.length; i++) {
					char temp = cc[i];
					for (char c = 'a'; c <= 'z'; c++) {
						cc[i] = c;
						String str = String.valueOf(cc);
						if (c != temp && wordSet.contains(str)) {
							if (str.equals(endWord))
								flag = true;
							if (!visitedThisLev.contains(str)) {
								List<String> one = new ArrayList<String>();
								one.add(cur);
								graph.put(str, one);
								queue.offer(str);
								visitedThisLev.add(str);
							} else {
								List<String> one = graph.get(str);
								one.add(cur);

							}
						}
					}
					cc[i] = temp;
				}
			}
			wordSet.removeAll(visitedThisLev);

			if (flag) {
				List<String> one = new LinkedList<>();
				one.add(endWord);
				search(res, one, endWord, beginWord, graph);
				return res;
			}
		}
		return res;
	}

	public static void search(List<List<String>> res, List<String> one, String start, String end,
			Map<String, List<String>> graph) {
		if (start.equals(end)) {
			res.add(new LinkedList<String>(one));
			return;
		}
		List<String> next = graph.get(start);
		for (String str : next) {
			one.add(0, str);
			search(res, one, str, end, graph);
			one.remove(0);
		}
	}

	public static int subarraysWithKDistinct(int[] A, int K) {
		if (A == null || A.length == 0)
			return 0;
		return atMost(A, K) - atMost(A, K - 1);
	}

	public static int atMost(int[] A, int K) {
		int res = 0;
		int count = 0;
		int n = A.length;
		Map<Integer, Integer> map = new HashMap<>();
		for (int i = 0, j = 0; j < n; j++) {
			if (count <= K) {
				map.put(A[j], map.getOrDefault(A[j], 0) + 1);
				if (map.get(A[j]) == 1)
					count++;
			}
			while (count > K) {
				map.put(A[i], map.get(A[i]) - 1);
				if (map.get(A[i]) == 0)
					count--;
				i++;
			}
			res += j - i + 1;
		}
		return res;
	}

	public int findMaxLength(int[] nums) {
		if (nums == null || nums.length == 0)
			return 0;
		int max = 0;
		int count = 0;
		int n = nums.length;
		Map<Integer, Integer> map = new HashMap<>();
		map.put(0, -1);
		for (int i = 0; i < n; i++) {
			if (nums[i] == 0) {
				count++;
			} else {
				count--;
			}
			if (map.containsKey(count)) {
				max = Math.max(max, i - map.get(count));
			} else {
				map.put(count, i);
			}
		}
		return max;
	}

	public int ladderLength(String beginWord, String endWord, List<String> wordList) {
		if (beginWord == null || endWord == null || wordList == null) {
			return 0;
		}
		if (beginWord == endWord)
			return 1;
		HashSet<String> wordSet = new HashSet<String>();
		for (String s : wordList) {
			wordSet.add(s);
		}
		Queue<String> queue = new LinkedList<>();
		queue.offer(beginWord);
		wordSet.remove(beginWord);
		int minDis = 2;
		while (!queue.isEmpty()) {
			int size = queue.size();
			while (size-- > 0) {
				String s = queue.poll();
				char[] ch = s.toCharArray();
				for (int i = 0; i < ch.length; i++) {
					char temp = ch[i];
					for (char c = 'a'; c <= 'z'; c++) {
						ch[i] = c;
						String str = String.valueOf(ch);
						if (c != temp && wordSet.contains(str)) {
							if (str.equals(endWord))
								return minDis;
							queue.offer(str);
							wordSet.remove(str);
						}
					}
					ch[i] = temp;
				}
			}
			minDis++;
		}
		return 0;
	}

	public List<List<Integer>> pacificAtlantic(int[][] matrix) {
		List<List<Integer>> res = new ArrayList<>();
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			return res;
		int row = matrix.length, col = matrix[0].length;
		Queue<Integer> queue = new LinkedList<>();
		boolean[][] pacific = new boolean[row][col];
		boolean[][] atlantic = new boolean[row][col];
		for (int j = 0; j < col; j++) {
			queue.offer(j);
			pacific[0][j] = true;
		}
		for (int i = 1; i < row; i++) {
			queue.offer(i * col);
			pacific[i][0] = true;
		}
		bfs(matrix, pacific, atlantic, queue, res);
		for (int j = 0; j < col; j++) {
			queue.offer((row - 1) * col + j);
			atlantic[row - 1][j] = true;
		}
		for (int i = 0; i < row - 1; i++) {
			queue.offer(i * col + col - 1);
			atlantic[i][col - 1] = true;
		}
		bfs(matrix, atlantic, pacific, queue, res);
		return res;
	}

	private void bfs(int[][] matrix, boolean[][] self, boolean[][] other, Queue<Integer> queue,
			List<List<Integer>> res) {
		int row = matrix.length;
		int col = matrix[0].length;
		int[][] directions = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
		while (!queue.isEmpty()) {
			int cur = queue.poll();
			int i = cur / col, j = cur % col;
			if (other[i][j]) {
				List<Integer> list = new ArrayList<>();
				list.add(i);
				list.add(j);
				res.add(list);
			}
			for (int[] dir : directions) {
				int ii = i + dir[0];
				int jj = j + dir[1];
				if (ii >= 0 && ii < row && jj >= 0 && jj < col && !self[ii][jj] && matrix[i][j] <= matrix[ii][jj]) {
					self[ii][jj] = true;
					queue.offer(ii * col + jj);
				}
			}
		}
	}

	public static int shortestDistance(int[][] grid) {
		int row = grid.length;
		int col = grid[0].length;
		int[][] sum = new int[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (grid[i][j] == 1) {
					bfs(sum, grid, i, j);
				}
			}
		}

		int min = Integer.MAX_VALUE;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (grid[i][j] == 0) {
					min = Math.min(min, sum[i][j]);
				}
			}
		}
		return min == Integer.MAX_VALUE ? -1 : min;
	}

	public static void bfs(int[][] sum, int[][] grid, int i, int j) {
		Queue<Integer> queue = new LinkedList<>();
		int row = grid.length;
		int col = grid[0].length;
		queue.offer(i * col + j);
		int minLength = 1;
		boolean[][] visited = new boolean[row][col];
		int[][] directions = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
		while (!queue.isEmpty()) {
			int size = queue.size();
			while (size-- > 0) {
				int cur = queue.poll();
				int curI = cur / col, curJ = cur % col;
				for (int[] dir : directions) {
					int ii = curI + dir[0];
					int jj = curJ + dir[1];
					if (ii >= 0 && ii < row && jj >= 0 && jj < col && grid[ii][jj] == 0 && !visited[ii][jj]) {
						queue.offer(ii * col + jj);
						sum[ii][jj] += minLength;
						visited[ii][jj] = true;
					}
				}
			}
			minLength++;
		}
		for (int m = 0; m < row; m++) {
			for (int n = 0; n < col; n++) {
				if (grid[m][n] == 0 && !visited[m][n]) {
					grid[m][n] = 2;
				}
			}
		}
	}

	public int[][] updateMatrix(int[][] matrix) {
		int row = matrix.length;
		int col = matrix[0].length;
		int[][] res = new int[row][col];
		Queue<int[]> queue = new LinkedList<>();
		int[][] directions = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
		addAllZeros2(queue, matrix);
		int minLength = 1;
		while (!queue.isEmpty()) {
			int size = queue.size();
			while (size-- > 0) {
				int[] temp = queue.poll();
				int i = temp[0], j = temp[1];
				for (int[] dir : directions) {
					int ii = i + dir[0];
					int jj = j + dir[1];
					if (ii >= 0 && ii < row && jj >= 0 && jj < col && matrix[ii][jj] == 1 && res[ii][jj] == 0) {
						res[ii][jj] = minLength;
						queue.add(new int[] { ii, jj });
					}
				}
			}
			minLength++;
		}
		return res;
	}

	public static void addAllZeros2(Queue<int[]> queue, int[][] rooms) {
		int row = rooms.length;
		int col = rooms[0].length;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (rooms[i][j] == 0) {
					queue.add(new int[] { i, j });
				}
			}
		}
	}

	public static void wallsAndGates(int[][] rooms) {
		if (rooms == null || rooms.length == 0 || rooms[0] == null || rooms[0].length == 0)
			return;
		Queue<Integer> queue = new LinkedList<>();
		addAllZeros(queue, rooms);
		int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };
		int row = rooms.length;
		int col = rooms[0].length;
		int minLength = 1;
		while (!queue.isEmpty()) {
			int size = queue.size();
			while (size-- > 0) {
				int cur = queue.poll();
				int i = cur / col, j = cur % col;
				for (int[] dir : directions) {
					int ii = i + dir[0];
					int jj = j + dir[1];
					if (ii >= 0 && ii < row && jj >= 0 && jj < col && rooms[ii][jj] == Integer.MAX_VALUE) {
						rooms[ii][jj] = minLength;
						queue.offer(ii * col + jj);
					}
				}
			}
			minLength++;
		}
	}

	public static void addAllZeros(Queue<Integer> queue, int[][] rooms) {
		int row = rooms.length;
		int col = rooms[0].length;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (rooms[i][j] == 0) {
					queue.add(i * col + j);
				}
			}
		}
	}

	public String frequencySort(String s) {
		char[] ch = s.toCharArray();
		StringBuilder sb = new StringBuilder();
		Map<Character, Integer> map = new HashMap<>();
		for (char c : ch) {
			map.put(c, map.getOrDefault(c, 0) + 1);
		}
		PriorityQueue<Character> pq = new PriorityQueue<>((a, b) -> map.get(b) - map.get(a));
		pq.addAll(map.keySet());
		while (!pq.isEmpty()) {
			char c = pq.poll();
			int k = map.get(c);
			for (int i = 0; i < k; i++) {
				sb.append(c);
			}
		}
		return sb.toString();
	}

	public int firstUniqChar(String s) {
		char[] ch = s.toCharArray();
		int len = s.length();
		int[] count = new int[256];
		for (int i = 0; i < len; i++) {
			count[ch[i]]++;
		}
		for (int i = 0; i < len; i++) {
			if (count[ch[i]] == 1) {
				return i;
			}
		}
		return -1;
	}

	public int lengthOfLongestSubstringKDistinct(String s, int k) {
		if (k == 0)
			return 0;
		char[] ch = s.toCharArray();
		int[] count = new int[256];
		int different = 0;
		int fast = 0;
		int n = s.length();
		int resL = 0, resR = 0;
		for (int i = 0; i < n; i++) {
			while (different <= k && fast < n) {
				count[ch[fast]]++;
				if (count[ch[fast]] == 1)
					different++;
				fast++;
			}
			if (different == k + 1) {
				if (resR - resL < fast - 1 - i) {
					resL = i;
					resR = fast - 1;
				}
			} else {
				if (resR - resL < fast - i) {
					resL = i;
					resR = fast;
				}
			}
			count[ch[i]]--;
			if (count[ch[i]] == 0) {
				different--;
			}
		}
		if (resR == 0)
			return s.length();
		return resR - resL;
	}

	public static int lengthOfLongestSubstringTwoDistinct(String s) {
		char[] ch = s.toCharArray();
		int[] count = new int[256];
		int different = 0;
		int fast = 0;
		int n = s.length();
		int resL = 0, resR = 0;
		for (int i = 0; i < n; i++) {
			while (different <= 2 && fast < n) {
				count[ch[fast]]++;
				if (count[ch[fast]] == 1)
					different++;
				fast++;
			}
			if (different == 3) {
				if (resR - resL < fast - 1 - i) {
					resL = i;
					resR = fast - 1;
				}
			} else {
				if (resR - resL < fast - i) {
					resL = i;
					resR = fast;
				}
			}
			count[ch[i]]--;
			if (count[ch[i]] == 0) {
				different--;
			}
		}
		if (resR == 0)
			return s.length();
		return resR - resL;
	}

	public static int[] maxSlidingWindow(int[] nums, int k) {
		if (nums == null || nums.length == 0 || k <= 0)
			return new int[0];
		if (k == 1)
			return nums;
		int[] res = new int[nums.length - k + 1];
		Deque<Integer> dq = new LinkedList<>();
		dq.offer(nums[0]);
		for (int i = 1; i < k; i++) {
			while (!dq.isEmpty() && nums[i] > dq.peekLast()) {
				dq.pollLast();
			}
			dq.offer(nums[i]);
		}
		int n = nums.length;
		res[0] = dq.peekFirst();
		for (int i = k; i < n; i++) {
			if (nums[i - k] == dq.peek()) {
				dq.removeFirst();
			}
			while (!dq.isEmpty() && nums[i] > dq.peekLast()) {
				dq.pollLast();
			}
			dq.offer(nums[i]);
			res[i - k + 1] = dq.peekFirst();
		}
		return res;
	}

	public String minWindow(String s, String t) {
		if (t.length() == 0)
			return "";
		char[] charS = s.toCharArray();
		char[] charT = t.toCharArray();
		int[] storeS = new int[256];
		int[] storeT = new int[256];
		int count = 0;
		for (char c : charT) {
			storeT[c]++;
			if (storeT[c] == 1)
				count++;
		}
		int fast = 0;
		int validNum = 0;
		int resL = -1, resR = -1;
		int len = s.length();
		for (int slow = 0; slow < len; slow++) {
			while (fast < len && validNum < count) {
				storeS[charS[fast]]++;
				if (storeS[charS[fast]] == storeT[charS[fast]]) {
					validNum++;
				}
				fast++;
			}
			if (validNum == count) {
				if (resL == -1 || (resR - resL) > fast - slow) {
					resL = slow;
					resR = fast;
				}
			}
			storeS[charS[slow]]--;
			if (storeS[charS[slow]] + 1 == storeT[charS[slow]])
				validNum--;
		}
		if (resL == -1)
			return "";
		return s.substring(resL, resR);
	}

	public static int minSubArrayLen(int s, int[] nums) {
		if (nums.length == 0)
			return 0;
		int fast = 0;
		int slow = 0;
		long min = nums.length + 1;
		int sum = nums[0];
		while (fast < nums.length) {
			if (sum >= s) {
				min = Math.min(min, fast - slow + 1);
				sum -= nums[slow++];
			} else {
				fast++;
				if (fast >= nums.length)
					break;
				sum += nums[fast];
			}
		}
		return min == nums.length + 1 ? 0 : (int) min;
	}

	public int maxSubArrayLen(int[] nums, int k) {
		if (nums.length == 0)
			return 0;
		int max = 0;
		int sum = 0;
		Map<Integer, Integer> map = new HashMap<>();
		map.put(0, -1);
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
			if (map.containsKey(sum - k)) {
				max = Math.max(max, i - map.get(sum - k));
			}
			if (!map.containsKey(sum)) {
				map.put(sum, i);
			}
		}
		return max;
	}

	public int subarraySum(int[] nums, int k) {
		int count = 0;
		Map<Integer, Integer> map = new HashMap<>();
		map.put(0, 1);
		int sum = 0;
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
			if (map.containsKey(sum - k)) {
				count += map.get(sum - k);
			}
			map.put(sum, map.getOrDefault(sum, 0) + 1);
		}
		return count;
	}

	public int totalNQueens(int n) {
		boolean[] dpUp = new boolean[n];
		boolean[] dpUpLeft = new boolean[2 * n - 1];
		boolean[] dpUpRight = new boolean[2 * n - 1];
		int[] count = new int[1];
		totalNQueens(n, count, 0, dpUp, dpUpLeft, dpUpRight);
		return count[0];
	}

	public void totalNQueens(int n, int[] count, int index, boolean[] dpUp, boolean[] dpUpLeft, boolean[] dpUpRight) {
		if (index == n) {
			count[0]++;
			return;
		}
		for (int i = 0; i < n; i++) {
			if (!dpUp[i] && !dpUpLeft[i - index + n - 1] && !dpUpRight[i + index]) {
				dpUp[i] = true;
				dpUpLeft[i - index + n - 1] = true;
				dpUpRight[i + index] = true;
				totalNQueens(n, count, index + 1, dpUp, dpUpLeft, dpUpRight);
				dpUp[i] = false;
				dpUpLeft[i - index + n - 1] = false;
				dpUpRight[i + index] = false;
			}
		}
	}

	public List<List<String>> solveNQueens(int n) {
		List<List<String>> res = new ArrayList<>();
		boolean[] dpUp = new boolean[n];
		boolean[] dpUpLeft = new boolean[2 * n - 1];
		boolean[] dpUpRight = new boolean[2 * n - 1];
		char[][] matrix = new char[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				matrix[i][j] = '.';
			}
		}
		solveNQueens(n, matrix, res, new ArrayList<>(), 0, dpUp, dpUpLeft, dpUpRight);
		return res;
	}

	public void solveNQueens(int n, char[][] matrix, List<List<String>> res, List<String> sofar, int index,
			boolean[] dpUp, boolean[] dpUpLeft, boolean[] dpUpRight) {
		if (index == n) {
			res.add(new ArrayList<>(sofar));
			return;
		}
		for (int i = 0; i < n; i++) {
			if (!dpUp[i] && !dpUpLeft[i - index + n - 1] && !dpUpRight[i + index]) {
				dpUp[i] = true;
				dpUpLeft[i - index + n - 1] = true;
				dpUpRight[i + index] = true;
				matrix[index][i] = 'Q';
				sofar.add(String.valueOf(matrix[index]));
				solveNQueens(n, matrix, res, sofar, index + 1, dpUp, dpUpLeft, dpUpRight);
				dpUp[i] = false;
				dpUpLeft[i - index + n - 1] = false;
				dpUpRight[i + index] = false;
				matrix[index][i] = '.';
				sofar.remove(sofar.size() - 1);
			}
		}
	}

	public List<List<Integer>> combine(int n, int k) {
		List<List<Integer>> res = new ArrayList<>();
		combine(res, new ArrayList<>(), n, k, 1);
		return res;
	}

	public void combine(List<List<Integer>> res, List<Integer> sofar, int n, int k, int index) {
		if (k == 0) {
			res.add(new ArrayList<Integer>(sofar));
			return;
		}
		for (int i = index; i <= n; i++) {
			if (n - i + 1 < k)
				return;
			sofar.add(i);
			combine(res, sofar, n, k - 1, i + 1);
			sofar.remove(sofar.size() - 1);
		}
	}

	public List<String> wordBreak2(String s, List<String> wordDict) {
		int n = s.length();
		List<String>[] dp = new LinkedList[n + 1];
		List<String> initial = new LinkedList<>();
		initial.add("");
		dp[0] = initial;
		for (int i = 1; i <= n; i++) {
			List<String> list = new LinkedList<>();
			for (int j = 0; j < i; j++) {
				if (dp[j].size() > 0 && wordDict.contains(s.substring(j, i))) {
					for (String ss : dp[j]) {
						list.add((ss.equals("") ? "" : ss + " ") + s.substring(j, i));
					}
				}
			}
			dp[i] = list;
		}
		return dp[n];
	}

	public boolean wordBreak(String s, List<String> wordDict) {
		int n = s.length();
		boolean[] dp = new boolean[n + 1];
		dp[0] = true;
		for (int i = 1; i <= n; i++) {
			for (int j = 0; j < i; j++) {
				if (dp[j] && wordDict.contains(s.substring(j, i))) {
					dp[i] = true;
					break;
				}
			}
		}
		return dp[n];
	}

	public static int lengthOfLIS1(int[] nums) {
		if (nums.length == 0)
			return 0;
		List<Integer> list = new ArrayList<>();
		list.add(nums[0]);
		int index = 0;
		for (int i = 1; i < nums.length; i++) {
			index = binarySearch(list, nums[i]) + 1;
			if (index <= list.size() - 1) {
				list.set(index, nums[i]);
			} else {
				list.add(nums[i]);
			}
		}
		return list.size();
	}

	public static int binarySearch(List<Integer> list, int target) {
		int left = 0;
		int right = list.size() - 1;
		int mid = 0;
		while (left < right - 1) {
			mid = (right - left) / 2 + left;
			if (list.get(mid) >= target) {
				right = mid;
			} else {
				left = mid;
			}
		}
		if (list.get(right) < target)
			return right;
		if (list.get(left) < target)
			return left;
		return left - 1;
	}

	public void rotate(int[][] matrix) {
		int len = matrix.length - 1;
		rotate(matrix, 0, len);
	}

	public void rotate(int[][] matrix, int index, int len) {
		if (len <= 0)
			return;
		int temp = 0;
		for (int i = 0; i < len; i++) {
			temp = matrix[index][index + i];
			matrix[index][index + i] = matrix[index + len - i][index];
			matrix[index + len - i][index] = matrix[index + len][index + len - i];
			matrix[index + len][index + len - i] = matrix[index + i][index + len];
			matrix[index + i][index + len] = temp;
		}
		rotate(matrix, index + 1, len - 2);
	}

//	Largest Rectangle in Histogram
	public static int largestRectangleArea(int[] heights) {
		int max = 0;
		int[] head = { -1, -1 };
		Stack<int[]> stack = new Stack<>();
		stack.push(head);
		int temp = 0;
		for (int i = 0; i < heights.length; i++) {
			if (heights[i] >= stack.peek()[1]) {
				stack.push(new int[] { i, heights[i] });
			} else {
				while (stack.peek()[1] > heights[i]) {
					temp = stack.pop()[1];
					max = Math.max(temp * (i - stack.peek()[0] - 1), max);
				}
				stack.push(new int[] { i, heights[i] });
			}
		}
		while (stack.peek()[1] > -1) {
			temp = stack.pop()[1];
			max = Math.max(temp * (heights.length - stack.peek()[0] - 1), max);
		}
		return max;
	}

//	Subsets 2
	public static List<List<Integer>> subsetsWithDup(int[] nums) {
		Arrays.sort(nums);
		List<List<Integer>> res = new ArrayList<>();
		subsetsWithDup(nums, res, new LinkedList<Integer>(), 0);
		return res;
	}

	public static void subsetsWithDup(int[] nums, List<List<Integer>> res, List<Integer> sofar, int index) {
		res.add(new ArrayList<Integer>(sofar));
		for (int i = index; i < nums.length; i++) {
			if (i > index && nums[i] == nums[i - 1]) {
				continue;
			}
			sofar.add(nums[i]);
			subsetsWithDup(nums, res, sofar, i + 1);
			sofar.remove(sofar.size() - 1);
		}
	}

//	Subsets 1
	public static List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		List<Integer> sofar = new ArrayList<>();
		subsetsHelper(nums, res, sofar, 0);
		return res;
	}

	public static void subsetsHelper(int[] nums, List<List<Integer>> res, List<Integer> sofar, int index) {
		res.add(new ArrayList<Integer>(sofar));
		for (int i = index; i < nums.length; i++) {
			sofar.add(nums[i]);
			subsetsHelper(nums, res, sofar, i + 1);
			sofar.remove(sofar.size() - 1);
		}
	}

	public List<Integer> preorderTraversal(TreeNode root) {
		List<Integer> list = new ArrayList<>();
		if (root == null)
			return list;
		preorderTraversalHelper(root, list);
		return list;
	}

	public void preorderTraversalHelper(TreeNode root, List<Integer> list) {
		if (root == null)
			return;
		list.add(root.val);
		preorderTraversalHelper(root.left, list);
		preorderTraversalHelper(root.right, list);
	}

	public TreeNode upsideDownBinaryTree(TreeNode root) {
		if (root == null || root.left == null)
			return root;
		TreeNode newRoot = upsideDownBinaryTree(root.left);
		root.left.left = root.right;
		root.left.right = root;
		root.left = null;
		root.right = null;
		return newRoot;
	}

	public TreeNode deleteNode(TreeNode root, int key) {
		if (root == null)
			return null;
		if (root.val == key) {
			if (root.left != null && root.right != null) {
				root.val = findMin(root.right).val;
				root.right = deleteNode(root.right, root.val);
			} else {
				return root.left == null ? root.right : root.left;
			}
		} else if (root.val > key) {
			root.left = deleteNode(root.left, key);
		} else {
			root.right = deleteNode(root.right, key);
		}
		return root;
	}

	public TreeNode findMin(TreeNode root) {
		while (root.left != null) {
			root = root.left;
		}
		return root;
	}

	public Node cloneGraph(Node node) {

		return new Node(0, null);
	}

//	public Node copyRandomList(Node head) {
//        Node cur1=head;
//        Node dummy=new Node(0,null,null);
//        Node cur2=dummy;
//        Map<Node,Node> map=new HashMap<>();
//        
//        while(cur1!=null) {
//        	if(!map.containsKey(cur1)) {
//        		map.put(cur1, new Node(cur1.val,null,null));
//        	}
//        	cur2.next=map.get(cur1);
//        	if(cur1.random!=null) {
//        		if(!map.containsKey(cur1.random)) {
//        			map.put(cur1.random, new Node(cur1.random.val,null,null));
//        		}
//        		cur2.next.random=map.get(cur1.random);
//        	}
//        	cur1=cur1.next;
//        	cur2=cur2.next;
//        }
//        return dummy.next;
//    }

	public int closestValue(TreeNode root, double target) {
		int res = root.val;
		TreeNode cur = root;
		while (cur != null) {
			if (Math.abs(cur.val - target) < 0.0000001)
				return cur.val;
			if (Math.abs(cur.val - target) < Math.abs(res - target))
				res = cur.val;
			if (cur.val > target) {
				cur = cur.left;
			} else {
				cur = cur.right;
			}
		}
		return res;
	}

	public static int combinationSum4(int[] nums, int target) {
		if (nums == null || nums.length == 0)
			return 0;
		Arrays.sort(nums);
		int[] res = new int[target + 1];
		res[0] = 1;
		for (int i = 1; i <= target; i++) {
			for (int num : nums) {
				if (num <= i) {
					res[i] += res[i - num];
				} else {
					break;
				}
			}
		}
		return res[target];
	}

	public List<List<Integer>> combinationSum3(int k, int n) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		combinationSum3Helper(k, n, result, new LinkedList<Integer>(), 1);
		return result;
	}

	public void combinationSum3Helper(int k, int n, List<List<Integer>> result, List<Integer> sofar, int level) {
		if (k == 0 && n == 0) {
			result.add(new ArrayList<Integer>(sofar));
			return;
		}
		if (k == 0)
			return;
		for (int i = level; i <= n; i++) {
			if (i == 10)
				return;
			int minSum = (i + i + k - 1) * k / 2;
			if (minSum > n)
				return;
			sofar.add(i);
			combinationSum3Helper(k - 1, n - i, result, sofar, i + 1);
			sofar.remove(sofar.size() - 1);
		}
	}

	public List<List<Integer>> combinationSum33(int k, int n) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		combinationSum3Helper(k, n, result, new LinkedList<Integer>(), 1);
		return result;
	}

	public void combinationSum33Helper(int k, int n, List<List<Integer>> result, List<Integer> sofar, int level) {
		if (k == 0 && n == 0) {
			result.add(new ArrayList<Integer>(sofar));
			return;
		}
		if (k == 0)
			return;
		for (int i = level; i <= n; i++) {
			int minSum = (i + i + k - 1) * k / 2;
			if (minSum > n)
				return;
			sofar.add(i);
			combinationSum3Helper(k - 1, n - i, result, sofar, i + 1);
			sofar.remove(sofar.size() - 1);
		}
	}

	public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
		List<List<Integer>> result = new ArrayList<>();
		if (candidates == null || candidates.length == 0) {
			return result;
		}
		Arrays.sort(candidates);
		combinationSum2Helper(candidates, target, result, 0, new LinkedList<Integer>());
		return result;
	}

	public static void combinationSum2Helper(int[] candidates, int target, List<List<Integer>> result, int index,
			LinkedList<Integer> temp) {
		if (target == 0) {
			result.add(new ArrayList<Integer>(temp));
			return;
		}
		for (int i = index; i < candidates.length; i++) {
			if (target < candidates[i]) {
				return;
			}
			if (i != index && candidates[i] == candidates[i - 1]) {
				continue;
			}
			temp.add(candidates[i]);
			combinationSum2Helper(candidates, target - candidates[i], result, i + 1, temp);
			temp.remove(temp.size() - 1);
		}
	}

	public List<List<Integer>> combinationSum(int[] candidates, int target) {
		List<List<Integer>> result = new ArrayList<>();
		combinationSumHelper(candidates, target, 0, result, new LinkedList<>());
		return result;
	}

	public void combinationSumHelper(int[] candidates, int target, int index, List<List<Integer>> result,
			LinkedList<Integer> temp) {
		if (target == 0) {
			result.add(new ArrayList<>(temp));
			return;
		}

		if (index >= candidates.length) {
			return;
		}

		if (candidates[index] <= target) {
			temp.add(candidates[index]);
			combinationSumHelper(candidates, target - candidates[index], index, result, temp);
			temp.removeLast();
		}

		combinationSumHelper(candidates, target, index + 1, result, temp);
	}

	public int numTrees(int n) {
		if (n <= 1)
			return n;
		int[] dp = new int[n + 1];
		dp[0] = 1;
		dp[1] = 1;
		for (int i = 2; i <= n; i++) {
			for (int j = 0; j < i; j++) {
				dp[i] += dp[j] * dp[i - j - 1];
			}
		}
		return dp[n];
	}

	public boolean hasCycle(ListNode head) {
		if (head == null)
			return false;
		ListNode slow = head;
		ListNode fast = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow)
				return true;
		}
		return false;
	}

	public ListNode detectCycle(ListNode head) {
		if (head == null)
			return head;
		ListNode slow = head;
		ListNode fast = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow) {
				ListNode slow2 = head;
				while (slow != slow2) {
					slow = slow.next;
					slow2 = slow2.next;
				}
				return slow;
			}
		}
		return null;
	}

	public ListNode reverseList(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode newHead = reverseList(head.next);
		head.next.next = head;
		head.next = null;
		return newHead;
	}

	public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sofar = new ArrayList<Integer>();
		sofar.add(0);
		allPathsSourceTarget(graph, 0, sofar, res);
		return res;
	}

	public void allPathsSourceTarget(int[][] graph, int level, List<Integer> sofar, List<List<Integer>> res) {
		if (level == graph.length - 1) {
			res.add(new ArrayList<Integer>(sofar));
			return;
		}
		for (int i = 0; i < graph[level].length; i++) {
			sofar.add(graph[level][i]);
			allPathsSourceTarget(graph, graph[level][i], sofar, res);
			sofar.remove(sofar.size() - 1);
		}
	}

	public boolean isSameTree(TreeNode p, TreeNode q) {
		if (p == null && q == null)
			return true;
		if (p == null || q == null)
			return false;
		if (p.val != q.val)
			return false;
		return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
	}

	public boolean isSymmetric(TreeNode root) {
		return root == null ? true : isSymmetric(root.left, root.right);
	}

	public boolean isSymmetric(TreeNode left, TreeNode right) {
		if (left == null && right == null)
			return true;
		if (left == null || right == null)
			return false;
		if (left.val != right.val)
			return false;
		return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
	}

	public TreeNode invertTree(TreeNode root) {
		if (root == null)
			return root;
		TreeNode left = invertTree(root.left);
		TreeNode right = invertTree(root.right);
		root.left = right;
		root.right = left;
		return root;
	}

	public static boolean isValidBST(TreeNode root) {
		return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
	}

	public static boolean isValidBST(TreeNode root, long min, long max) {
		if (root == null)
			return true;
		int value = root.val;
		if (value >= max || value <= min)
			return false;
		return isValidBST(root.left, min, value) && isValidBST(root.right, value, max);
	}

	public List<Integer> inorderTraversal(TreeNode root) {
		List<Integer> list = new LinkedList<Integer>();
		if (root == null)
			return list;
		inorderTraversalHelper(root, list);
		return list;
	}

	public void inorderTraversalHelper(TreeNode root, List<Integer> list) {
		if (root == null)
			return;
		inorderTraversalHelper(root.left, list);
		list.add(root.val);
		inorderTraversalHelper(root.right, list);
	}

	public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
		List<List<Integer>> result = new LinkedList<List<Integer>>();
		if (root == null)
			return result;
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		queue.offer(root);
		int level = 0;
		while (!queue.isEmpty()) {
			List<Integer> sofar = new LinkedList<Integer>();
			int size = queue.size();
			level++;
			for (int i = 0; i < size; i++) {
				TreeNode current = queue.poll();
				sofar.add(current.val);
				if (current.left != null)
					queue.offer(current.left);
				if (current.right != null)
					queue.offer(current.right);
			}
			if (level % 2 != 0)
				Collections.reverse(sofar);
			result.add(sofar);
		}
		return result;
	}

	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> result = new LinkedList<List<Integer>>();
		if (root == null)
			return result;
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		queue.offer(root);
		int level = 0;
		while (!queue.isEmpty()) {
			List<Integer> sofar = new LinkedList<Integer>();
			int size = queue.size();
			level++;
			for (int i = 0; i < size; i++) {
				TreeNode current = queue.poll();
				sofar.add(current.val);
				if (current.left != null)
					queue.offer(current.left);
				if (current.right != null)
					queue.offer(current.right);
			}
			result.add(sofar);
		}
		return result;
	}

	public static int minDepth(TreeNode root) {
		if (root == null)
			return 0;
		int left = minDepth(root.left);
		int right = minDepth(root.right);
		if (left != 0 && right != 0) {
			return Math.min(left, right) + 1;
		} else if (left == 0) {
			return right + 1;
		} else {
			return left + 1;
		}
	}

	public static int maxDepth(TreeNode root) {
		if (root == null)
			return 0;
		int left = maxDepth(root.left);
		int right = maxDepth(root.right);
		return Math.max(left, right) + 1;
	}

	public static boolean isBalanced(TreeNode root) {
		if (root == null)
			return true;
		return helper(root) != -1 ? true : false;
	}

	public static int helper(TreeNode root) {
		if (root == null)
			return 0;
		int left = helper(root.left);
		if (left == -1)
			return -1;
		int right = helper(root.right);
		if (right == -1 || Math.abs(left - right) > 1)
			return -1;
		return Math.max(left, right) + 1;
	}

	public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
		int val = root.val;
		int pVal = p.val;
		int qVal = q.val;
		if (p.val > root.val && q.val > root.val) {
			return lowestCommonAncestorBST(root.right, p, q);
		} else if (p.val < root.val && q.val < root.val) {
			return lowestCommonAncestorBST(root.left, p, q);
		} else {
			return root;
		}
	}

	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		if (root == null || root == p || root == q)
			return root;
		TreeNode left = lowestCommonAncestor(root.left, p, q);
		TreeNode right = lowestCommonAncestor(root.right, p, q);
		if (left != null && right != null)
			return root;
		return left == null ? right : left;
	}

	public static TreeNode flatten(TreeNode root) {
		return flatten(root, null);
	}

	private static TreeNode flatten(TreeNode root, TreeNode tail) {
		if (root == null)
			return tail;
		TreeNode right = flatten(root.right, tail);
		TreeNode left = flatten(root.left, right);
		root.right = left;
		root.left = null;
		return root;
	}

	public static String largestNumber(int[] nums) {
		String[] str = new String[nums.length];
		for (int i = 0; i < nums.length; i++) {
			str[i] = String.valueOf(nums[i]);
		}
		Arrays.sort(str, new Comparator<String>() {

			@Override
			public int compare(String o1, String o2) {
				String s1 = o1 + o2;
				String s2 = o2 + o1;
				return s2.compareTo(s1);
			}
		});

		String res = "";
		for (String s : str) {
			res += s;
		}
		if (res.charAt(0) == '0')
			return "0";
		return res;
	}

	public static ListNode reverseBetween(ListNode head, int m, int n) {
		if (head == null || head.next == null || m == n)
			return head;
		int count = 1;
		ListNode before = new ListNode(0);
		before.next = head;
		ListNode cur = head;
		ListNode prev = null;
		ListNode next = null;
		while (count < m) {
			before = cur;
			cur = cur.next;
			count++;
		}
		next = cur.next;
		cur.next = prev;
		prev = cur;
		cur = next;
		ListNode tail = prev;
		count++;
		while (count <= n) {
			next = cur.next;
			cur.next = prev;
			prev = cur;
			cur = next;
			count++;
		}
		tail.next = cur;
		before.next = prev;
		return m != 1 ? head : prev;
	}

	public static int numIslands(char[][] grid) {
		if (grid == null || grid.length == 0 || grid[0] == null || grid[0].length == 0)
			return 0;
		int m = grid.length;
		int n = grid[0].length;
		int count = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (grid[i][j] == '1') {
					count++;
					dfs2(grid, i, j);
				}
			}
		}
		return count;
	}

	public static void dfs2(char[][] grid, int i, int j) {
		if (i < 0 || i > grid.length - 1 || j < 0 || j > grid[0].length - 1)
			return;
		if (grid[i][j] == '1') {
			grid[i][j] = '0';
			dfs2(grid, i + 1, j);
			dfs2(grid, i - 1, j);
			dfs2(grid, i, j + 1);
			dfs2(grid, i, j - 1);
		}
	}

	public static void solve(char[][] board) {
		if (board == null || board.length == 0 || board[0] == null || board[0].length == 0)
			return;
		int m = board.length;
		int n = board[0].length;
		for (int i = 0; i < m - 1; i++) {
			if (board[i][0] == 'O') {
				dfs(board, i, 0);
			}
		}
		for (int j = 0; j < n - 1; j++) {
			if (board[m - 1][j] == 'O') {
				dfs(board, m - 1, j);
			}
		}
		for (int i = m - 1; i > 0; i--) {
			if (board[i][n - 1] == 'O') {
				dfs(board, i, n - 1);
			}
		}
		for (int j = n - 1; j > 0; j--) {
			if (board[0][j] == 'O') {
				dfs(board, 0, j);
			}
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'O') {
					board[i][j] = 'X';
				}
			}
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'Y') {
					board[i][j] = 'O';
				}
			}
		}
	}

	public static void dfs(char[][] board, int i, int j) {
		if (i < 0 || i > board.length - 1 || j < 0 || j > board[0].length - 1)
			return;
		if (board[i][j] == 'O') {
			board[i][j] = 'Y';
			dfs(board, i + 1, j);
			dfs(board, i - 1, j);
			dfs(board, i, j + 1);
			dfs(board, i, j - 1);
		}
	}

	public static int minPathSum2(int[][] grid) {
		int m = grid.length;
		int n = grid[0].length;
		int[][] dp = new int[m][n];
		dp[0][0] = grid[0][0];
		for (int i = 1; i < m; i++) {
			dp[i][0] = dp[i - 1][0] + grid[i][0];
		}
		for (int j = 1; j < n; j++) {
			dp[0][j] = dp[0][j - 1] + grid[0][j];
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
			}
		}
		return dp[m - 1][n - 1];

		/*
		 * O(n) space complexity
		 */
//        int[] dp = new int[grid[0].length];
//		Arrays.fill(dp, Integer.MAX_VALUE);
//		dp[0] = 0;
//		for(int i = 0; i < grid.length; i++){
//		   dp[0] += grid[i][0];
//			for(int j = 1; j < grid[0].length; j++){
//				dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];		
//			}
//		}
//		return dp[grid[0].length - 1];
	}

	public static int uniquePaths(int m, int n) {
		if (m <= 0 || n <= 0)
			return 0;
		int[][] dp = new int[m][n];
		for (int i = 0; i < m; i++) {
			dp[i][0] = 1;
		}
		for (int j = 0; j < n; j++) {
			dp[0][j] = 1;
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}
		return dp[m - 1][n - 1];
	}

	public static void eightQueen(int[][] matrix, List<Integer> sol, int level, List<List<Integer>> res, boolean[] dpUp,
			boolean[] dpUpLeft, boolean[] dpUpRight) {
		if (level == matrix.length) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}

		for (int i = 0; i < matrix[0].length; i++) {
			if (isValid(sol, level, i, dpUp, dpUpLeft, dpUpRight)) {
				sol.add(i);
				dpUp[i] = true;
				dpUpLeft[i - level + 7] = true;
				dpUpRight[i + level] = true;
				eightQueen(matrix, sol, level + 1, res, dpUp, dpUpLeft, dpUpRight);
				sol.remove(sol.size() - 1);
				dpUp[i] = false;
				dpUpLeft[i - level + 7] = false;
				dpUpRight[i + level] = false;
			}
		}
	}

	public static boolean isValid(List<Integer> sol, int level, int i, boolean[] dpUp, boolean[] dpUpLeft,
			boolean[] dpUpRight) {
		if (dpUp[i] || dpUpLeft[i - level + 7] || dpUpRight[i + level])
			return false;
		return true;
	}

	public static void dfs22(int[][] matrix, List<Integer> sol, int level, List<List<Integer>> res) {
		if (level == matrix.length) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}

		for (int i = 0; i < matrix[0].length; i++) {
			if (isValid2(sol, level, i)) {
				sol.add(i);
				dfs22(matrix, sol, level + 1, res);
				sol.remove(sol.size() - 1);
			}
		}
	}

	public static boolean isValid2(List<Integer> sol, int level, int i) {
		for (int j = 0; j < level; j++) {
			if (sol.get(j) == i)
				return false;
		}
		for (int j = level - 1; j >= 0; j--) {
			if (sol.get(j) == i - level + j)
				return false;
		}
		for (int j = level - 1; j >= 0; j--) {
			if (sol.get(j) == i + level - j)
				return false;
		}
		return true;
	}

	public static void dfs2(char[] array, int index, List<String> res) {
//		base case
		if (index == array.length - 1) {
			res.add(new String(array));
		}
		for (int i = index; i < array.length; i++) {
			swap(array, index, i);
			dfs2(array, index + 1, res);
			swap(array, index, i);
		}
	}

	public static void swap(char[] array, int i, int j) {
		char temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}

	public static void dfs(char[] array, int index, StringBuilder sb, List<String> res) {
		res.add(sb.toString());
		for (int i = index; i < array.length; i++) {
			sb.append(array[i]);
			dfs(array, i + 1, sb, res);
			sb.deleteCharAt(sb.length() - 1);
		}
	}

	public static int lengthOfLIS(int[] nums) {

		return 0;
	}

	public static int firstBadVersion(int n) {
		if (n == 1)
			return 1;
		int left = 1;
		int right = n;
		int mid, res;
		while (left < right - 1) {
			mid = left + (right - left) / 2;
//			if(isBadVersion(mid)) {
//				right=mid;
//			}else {
//				left=mid;
//			}
		}
//		if(isBadVersion(left))return left;
		return right;
	}

	public static int findPeakElement(int[] nums) {
		if (nums.length == 1)
			return nums[0];
		if (nums[0] > nums[1])
			return nums[0];
		if (nums[nums.length - 1] > nums[nums.length - 2])
			return nums[nums.length - 1];
		int left = 0;
		int right = nums.length - 1;
		int mid, res;
		while (left < right - 1) {
			mid = left + (right - left) / 2;
			if (nums[mid] > nums[mid - 1] && nums[mid] < nums[mid + 1]) {
				return mid;
			} else if (nums[mid] < nums[mid] - 1) {
				right = mid;
			} else {
				left = mid;
			}
		}
		if (nums[left] > nums[right])
			return left;
		return -1;
	}

	public static int hIndex(int[] citations) {
		if (citations == null || citations.length == 0)
			return 0;
		int left = 0;
		int right = citations.length;
		int mid, res;
		while (left + 1 < right) {
			mid = left + (right - left) / 2;
			res = citations.length - mid;
			if (citations[res] >= mid) {
				left = mid;
			} else {
				right = mid;
			}
		}
		if (citations[citations.length - right] >= right)
			return right;
		return left;
	}

	public static boolean search2(int[] nums, int target) {
		if (nums == null || nums.length == 0)
			return false;

		int left = 0;
		int right = nums.length - 1;
		while (left <= right) {
			int mid = left + (right - left) / 2;

			if (nums[mid] == target)
				return true;
			else if (nums[left] < nums[mid]) {
				// left part sorted
				if (nums[left] <= target && target < nums[mid])
					right = mid - 1;
				else
					left = mid + 1;
			} else if (nums[left] > nums[mid]) {
				// right part sorted
				if (nums[mid] < target && target <= nums[right])
					left = mid + 1;
				else
					right = mid - 1;
			} else {
				// can not know which part is sorted directly, move left to right until
				// nums[left] != nums[mid]
				while (left < right && nums[left] == nums[mid]) {
					left++;
				}
				if (left < mid)
					continue;
				else
					left = mid + 1;
			}

		}
		return false;
	}

	public static boolean searchMatrix(int[][] matrix, int target) {
		if (matrix == null || matrix.length == 0 || matrix[0] == null || matrix[0].length == 0)
			return false;
		int l = 0, r = matrix.length - 1;
		int mid, res;
		while (l < r - 1) {
			mid = l + (r - l) / 2;
			if (matrix[mid][0] == target) {
				return true;
			} else if (matrix[mid][0] > target) {
				r = mid;
			} else {
				l = mid;
			}
		}
		if (matrix[r][0] == target || matrix[l][0] == target)
			return true;
		if (matrix[r][0] < target) {
			res = r;
		} else {
			res = l;
		}
		l = 0;
		r = matrix[0].length - 1;
		while (l <= r) {
			mid = l + (r - l) / 2;
			if (matrix[res][mid] == target) {
				return true;
			} else if (matrix[res][mid] > target) {
				r = mid - 1;
			} else {
				l = mid + 1;
			}
		}
		return false;
	}

	public static int mySqrt(int x) {
		if (x <= 1)
			return x;
		long l = 1, r = x / 2;
		long mid, res;
		while (l <= r) {
			mid = l + (r - l) / 2;
			res = mid * mid;
			if (res == x || res < x && (mid + 1) * (mid + 1) > x)
				return (int) mid;
			if (res < x) {
				l = mid + 1;
			} else {
				r = mid - 1;
			}
		}
		return 0;
	}

	public static int searchInsert(int[] nums, int target) {
		if (nums == null || nums.length == 0)
			return 0;
		int left = 0;
		int right = nums.length - 1;
		int mid = 0;
		while (left < right - 1) {
			mid = left + (right - left) / 2;
			if (nums[mid] == target) {
				return mid;
			} else if (nums[mid] < target) {
				left = mid;
			} else {
				right = mid;
			}
		}
		if (nums[left] >= target)
			return 0;
		if (nums[right] < target)
			return nums.length;
		return right;
	}

	public static int[] searchRange(int[] nums, int target) {
		if (nums == null || nums.length == 0)
			return new int[] { -1, -1 };
		int l = 0, r = nums.length - 1;
		int mid;
		while (l < r - 1) {
			mid = l + (r - l) / 2;
			if (nums[mid] < target) {
				l = mid;
			} else {
				r = mid;
			}
		}
		int resL, resR;
		if (nums[r] == target) {
			if (nums[l] == target) {
				resL = l;
			} else {
				resL = r;
			}
		} else if (nums[l] == target) {
			return new int[] { l, l };
		} else {
			return new int[] { -1, -1 };
		}

		l = 0;
		r = nums.length - 1;
		while (l < r - 1) {
			mid = l + (r - l) / 2;
			if (nums[mid] > target) {
				r = mid;
			} else {
				l = mid;
			}
		}
		if (nums[r] == target) {
			resR = r;
		} else {
			resR = l;
		}

		return new int[] { resL, resR };
	}

	public static int search(int[] nums, int target) {
		if (nums == null || nums.length == 0) {
			return -1;
		}
		int end = nums.length - 1;
		int l = 0, r = end, mid;
		while (l < r - 1) {
			mid = l + (r - l) / 2;
			if (nums[mid] > nums[0]) {
				l = mid;
			} else {
				r = mid;
			}
		}
		int begin;

		if (nums[l] <= nums[r]) {
			begin = 0;
		} else {
			begin = r;
		}

		l = 0;
		r = end;
		int realMid;
		while (l <= r) {
			mid = l + (r - l) / 2;
			realMid = (mid + begin) % nums.length;
			if (nums[realMid] == target) {
				return realMid;
			} else if (nums[realMid] < target) {
				l = mid + 1;
			} else {
				r = mid - 1;
			}
		}
		return -1;
	}

	public static String find(char[] ch, String s) {
		char[] sc = s.toCharArray();
		int[] store1 = new int[256];
		int[] store2 = new int[256];
		int k = 0;
		for (char c : ch) {
			store1[c]++;
			if (store1[c] == 1)
				k++;
		}

		int n = 0, j = 0;
		int resL = -1, resR = -1;
		int len = sc.length;
		for (int i = 0; i < len; i++) {
			while (j < len && n < k) {
				store2[sc[j]]++;
				if (store2[sc[j]] == store1[sc[j]]) {
					n++;
				}
				j++;
			}
			if (n == k) {
				if (resL == -1 || (resR - resL) > j - i) {
					resR = j;
					resL = i;
				}
			}
			store2[sc[i]]--;
			if (store1[sc[i]] - 1 == store2[sc[i]]) {
				n--;
			}
		}
		if (resL == -1) {
			return "";
		}
		return s.substring(resL, resR);
	}

	public int getLength(int[] arr) {
		if (arr == null || arr.length <= 1)
			return 0;
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int n = arr.length;
		int max = Integer.MAX_VALUE;
		for (int i = 0; i < n; i++) {
			if (map.containsKey(arr[i])) {
				max = Math.min(max, i - map.get(arr[i]) + 1);
			}
			map.put(arr[i], i);
		}
		return max;
	}

	public static int jump2(int[] nums) {
		if (nums == null || nums.length == 0)
			return 0;
		int max = 0;
		int curMax = 0;
		int res = 0;
		int n = nums.length - 1;
		for (int i = 0; i < n; i++) {
			if (i > curMax) {
				curMax = max;
				res++;
			}
			max = Math.max(max, i + nums[i]);
			if (max >= n)
				return res + 1;
		}
		return res;
	}

	public static ArrayList<Integer> divideAndMerge(ArrayList<Integer> array, int left, int right) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (left == right) {
			result.add(array.get(left));
			return result;
		}
		int mid = left + (right - left) / 2;
		ArrayList<Integer> leftResult = divideAndMerge(array, left, mid);
		ArrayList<Integer> rightResult = divideAndMerge(array, mid + 1, right);
		return merge(leftResult, rightResult);
	}

	public static ArrayList<Integer> merge(ArrayList<Integer> leftResult, ArrayList<Integer> rightResult) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		int i = 0, j = 0;
		while (i < leftResult.size() && j < rightResult.size()) {
			if (leftResult.get(i) <= rightResult.get(j)) {
				list.add(leftResult.get(i));
				i++;
			} else {
				list.add(rightResult.get(j));
				j++;
			}

			while (i < leftResult.size()) {
				list.add(leftResult.get(i));
				i++;
			}
			while (j < rightResult.size()) {
				list.add(rightResult.get(j));
				j++;
			}
		}
		return list;
	}

	public static ListNode reverse(ListNode lists) {
		ListNode head = lists;
		ListNode cur = head.next;
		ListNode tail = null;
		if (cur != null) {
			tail = cur.next;
		} else {
			return head;
		}
		head.next = null;
		while (tail != null) {
			cur.next = head;
			head = cur;
			cur = tail;
			tail = tail.next;
		}
		cur.next = head;
		return cur;
	}

	public static ListNode mergeKLists1(ListNode[] lists) {
		if (lists == null || lists.length == 0)
			return null;
		PriorityQueue<ListNode> queue = new PriorityQueue<>(lists.length, (a, b) -> a.val - b.val);
		ListNode dummy = new ListNode(0);
		ListNode cur = dummy;

		for (ListNode list : lists) {
			if (list != null) {
				queue.add(list);
			}
		}

		while (!queue.isEmpty()) {
			cur.next = queue.poll();
			cur = cur.next;
			if (cur.next != null) {
				queue.add(cur.next);
			}
		}

		return dummy.next;
	}

	public static ListNode mergeKLists(ListNode[] lists) {
		if (lists == null || lists.length == 0)
			return null;
		ListNode res = new ListNode(0);
		return sort(lists, 0, lists.length - 1);
	}

	public static ListNode sort(ListNode[] lists, int lo, int hi) {
		if (lo >= hi)
			return lists[lo];
		int mid = (hi - lo) / 2 + lo;
		ListNode l1 = sort(lists, lo, mid);
		ListNode l2 = sort(lists, mid + 1, hi);
		return merge(l1, l2);
	}

	public static ListNode merge(ListNode l1, ListNode l2) {
		if (l1 == null)
			return l2;
		if (l2 == null)
			return l1;
		if (l1.val < l2.val) {
			l1.next = merge(l1.next, l2);
			return l1;
		} else {
			l2.next = merge(l2.next, l1);
			return l2;
		}
	}

	public static List<String> generateParenthesis(int n) {
		List<String> list = new ArrayList<String>();
		backtrack(list, new StringBuilder(), 0, 0, n);
		return list;
	}

	public static void backtrack(List<String> list, StringBuilder str, int open, int close, int max) {

		if (str.length() == max * 2) {
			list.add(str.toString());
			return;
		}

		if (open < max) {
			backtrack(list, str.append("("), open + 1, close, max);
			str.deleteCharAt(str.length() - 1);
		}
		if (close < open) {
			backtrack(list, str.append(")"), open, close + 1, max);
			str.deleteCharAt(str.length() - 1);
		}
	}

	public static boolean isValid(String s) {
		Stack<Character> st = new Stack();
		char[] ch = s.toCharArray();
		int n = ch.length;
		for (int i = 0; i < n; i++) {
			if (ch[i] == '(') {
				st.push(')');
			} else if (ch[i] == '{') {
				st.push('}');
			} else if (ch[i] == '[') {
				st.push(']');
			} else if (st.empty()) {
				return false;
			} else {
				if (ch[i] == st.pop()) {
					continue;
				} else {
					return false;
				}
			}
		}
		return st.empty();
	}

	public static List<String> letterCombinations(String digits) {
		List<String> list = new ArrayList<>();
		if (digits.length() == 0) {
			return list;
		}
		String[] s = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
		dfs(s, digits, list, new StringBuilder());
		return list;
	}

	public static void dfs(String[] s, String digits, List<String> list, StringBuilder sb) {
		if (sb.length() == digits.length()) {
			list.add(sb.toString());
			return;
		}
		for (char ch : s[digits.charAt(sb.length()) - '0'].toCharArray()) {
			sb.append(ch);
			dfs(s, digits, list, sb);
			sb.deleteCharAt(sb.length() - 1);
		}
	}

	public static List<List<Integer>> threeSum(int[] nums) {
		Arrays.sort(nums);
		int n = nums.length;
		List<List<Integer>> resList = new ArrayList<>();
		int res, l, r;
		for (int i = 0; i < n - 2; i++) {
			if (i > 0 && nums[i] == nums[i - 1]) {
				continue;
			}
			l = i + 1;
			r = n - 1;
			res = 0 - nums[i];
			while (l < r) {
				if (nums[l] + nums[r] == res) {
					resList.add(Arrays.asList(nums[i], nums[l], nums[r]));
					while (l < r && nums[l] == nums[l + 1])
						l++;
					while (l < r && nums[r] == nums[r - 1])
						r--;
					l++;
					r--;
				} else if (nums[l] + nums[r] > res) {
					r--;
				} else {
					l++;
				}

			}
		}
		return resList;
	}

	public static int maxArea(int[] height) {
		int l = 0, r = height.length - 1;
		int res = 0, min;
		while (l < r) {
			min = Math.min(height[l], height[r]);
			res = Math.max(res, (r - l) * min);
			if (height[l] < height[r]) {
				l++;
			} else {
				r--;
			}
		}
		return res;
	}

	public static boolean isMatch(String s, String p) {
		int m = s.length();
		int n = p.length();
		boolean[][] dp = new boolean[m + 1][n + 1];
		dp[0][0] = true;
		for (int j = 0; j < n; j++) {
			if (p.charAt(j) == '*' && dp[0][j - 1]) {
				dp[0][j + 1] = true;
			}
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (p.charAt(j) == s.charAt(i) || p.charAt(j) == '.') {
					dp[i + 1][j + 1] = dp[i][j];
				} else if (p.charAt(j) == '*') {
					if (p.charAt(j - 1) == s.charAt(i) || p.charAt(j - 1) == '.') {
						dp[i + 1][j + 1] = dp[i][j + 1] || dp[i + 1][j - 1];
					} else {
						dp[i + 1][j + 1] = dp[i + 1][j - 1];
					}

				}

			}
		}

		return dp[m][n];
	}

	public static boolean isPalindrome(int x) {
		if (x < 0 || x != 0 && x % 10 == 0) {
			return false;
		}
		int res = 0;
		while (x > res) {
			res = res * 10 + x % 10;
			x /= 10;
		}
		return (x == res || x == res / 10);
	}

	public static int reverse2(int x) {
		int res = 0;
		if (x > 0) {
			while (x != 0 && (res < Integer.MAX_VALUE / 10 || res == Integer.MAX_VALUE / 10 && x % 10 <= 7)) {
				res = res * 10 + x % 10;
				x /= 10;
			}
		} else {
			while (x != 0 && (res > Integer.MIN_VALUE / 10 || res == Integer.MIN_VALUE / 10 && x % 10 >= -8)) {
				res = res * 10 + x % 10;
				x /= 10;
			}
		}
		if (x == 0) {
			return res;
		}
		return 0;
	}

	public static int lengthOfLongestSubstring(String s) {
		if (s.length() == 0) {
			return 0;
		}
		Set<Character> set = new HashSet<Character>();
		int i = 0, j = 0, max = 0;
		while (j < s.length()) {
			if (!set.contains(s.charAt(j))) {
				max = Math.max(max, j - i + 1);
				set.add(s.charAt(j++));
			} else {
				set.remove(s.charAt(i++));
			}
		}
		return max;
	}

	public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		ListNode res = new ListNode(1);
		ListNode a = res;
		int m = 0;
		int temp;
		while (l1 != null && l2 != null) {
			temp = l1.val + l2.val + m;
			if (temp >= 10) {
				a.next = new ListNode(temp - 10);
				m = 1;
			} else {
				a.next = new ListNode(temp);
				m = 0;
			}
			a = a.next;
			l1 = l1.next;
			l2 = l2.next;
		}
		if (l1 == null) {
			while (l2 != null) {
				temp = l2.val + m;
				if (temp >= 10) {
					a.next = new ListNode(temp - 10);
					m = 1;
				} else {
					a.next = new ListNode(temp);
					m = 0;
				}
				a = a.next;
				l2 = l2.next;
			}
		} else {
			while (l1 != null) {
				temp = l1.val + m;
				if (temp >= 10) {
					a.next = new ListNode(temp - 10);
					m = 1;
				} else {
					a.next = new ListNode(temp);
					m = 0;
				}
				a = a.next;
				l1 = l1.next;
			}
		}
		if (m == 1) {
			a.next = new ListNode(1);
		}
		return res.next;
	}

	public static int[] twoSum(int[] nums, int target) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < nums.length; i++) {
			if (map.containsKey(nums[i])) {
				return new int[] { map.get(nums[i]), i };
			}
			map.put(target - nums[i], i);
		}
		return new int[2];
	}

	public static boolean isNumber(String s) {
		int len = s.length();
		char cur;
		boolean flag = false, flag1 = false, flag2 = false;
		if (s.length() == 0) {
			return false;
		}
		if (s.length() == 1) {
			cur = s.charAt(0);
			return cur - '0' >= 0 && cur - '0' <= 9 ? true : false;
		}
		for (int i = 0; i < len - 1; i++) {
			cur = s.charAt(i);
			if (((cur == '+' || cur == '-'))) {
				if (i == 0 || s.charAt(i - 1) == 'e') {
					if ((s.charAt(i + 1) - '0' > 9 || s.charAt(i + 1) - '0' < 0)) {
						return false;
					}
				} else {
					return false;
				}
			} else if (cur == 'e') {
				if (flag || i == 0) {
					return false;
				}
				flag = true;
				flag1 = true;
				flag2 = false;
			} else if (cur == '.') {
				if (flag1) {
					return false;
				}
				if ((s.charAt(i + 1) - '0' > 9 || s.charAt(i + 1) - '0' < 0)) {
					return false;
				}
				flag1 = true;
			} else if ((cur - '0' > 9 || cur - '0' < 0)) {
				return false;
			}
		}
		return (s.charAt(len - 1) - '0' <= 9 && s.charAt(len - 1) - '0' >= 0) ? true : false;
	}

	public static String addBinary(String a, String b) {
		int l1 = a.length() - 1;
		int l2 = b.length() - 1;
		int a1, b1, res = 0, flag = 0;
		StringBuilder s = new StringBuilder();
		while (l1 >= 0 && l2 >= 0) {
			a1 = a.charAt(l1) - '0';
			b1 = b.charAt(l2) - '0';
			if (a1 + b1 + flag == 0) {
				res = 0;
				flag = 0;
			} else if (a1 + b1 + flag == 1) {
				res = 1;
				flag = 0;
			} else if (a1 + b1 + flag == 2) {
				res = 0;
				flag = 1;
			} else {
				res = 1;
				flag = 1;
			}
			s.insert(0, res);
			l1--;
			l2--;
		}
		while (l1 >= 0) {
			a1 = a.charAt(l1) - '0';
			if (a1 + flag == 1) {
				res = 1;
				flag = 0;
			} else if (a1 + flag == 2) {
				res = 0;
				flag = 1;
			} else {
				res = 0;
				flag = 0;
			}
			s.insert(0, res);
			l1--;
		}
		while (l2 >= 0) {
			b1 = b.charAt(l2) - '0';
			if (b1 + flag == 1) {
				res = 1;
				flag = 0;
			} else if (b1 + flag == 2) {
				res = 0;
				flag = 1;
			} else {
				res = 0;
				flag = 0;
			}
			s.insert(0, res);
			l2--;
		}
		if (flag == 1) {
			s.insert(0, 1);
		}
		return s.toString();
	}

	public static int minPathSum(int[][] grid) {
		int m = grid.length;
		int n = grid[0].length;
		int[][] opt = new int[m][n];
		opt[0][0] = grid[0][0];
		for (int j = 1; j < n; j++) {
			opt[0][j] = opt[0][j - 1] + grid[0][j];
		}
		for (int i = 1; i < m; i++) {
			opt[i][0] = opt[i - 1][0] + grid[i][0];
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				opt[i][j] = Math.min(opt[i - 1][j], opt[i][j - 1]) + grid[i][j];
			}
		}
		return opt[m - 1][n - 1];
	}

	private class IntervalComparator implements Comparator<Interval> {
		@Override
		public int compare(Interval o1, Interval o2) {
			return o1.start - o2.start;
		}
	}

	public static List<Interval> merge(List<Interval> interval) {
		List<Interval> list = new ArrayList<>();
		if (interval.size() == 0) {
			return list;
		}
		Collections.sort(interval, new Solution().new IntervalComparator());
		System.out.println(interval);
		int i1 = 0, i2 = 0;
		for (int i = 1; i < interval.size(); i++) {
			if (interval.get(i2).end >= interval.get(i).start) {
				if (interval.get(i).end > interval.get(i2).end) {
					i2 = i;
				}
			} else {
				list.add(new Interval(interval.get(i1).start, interval.get(i2).end));
				i1 = i;
				i2 = i;
			}
		}
		list.add(new Interval(interval.get(i1).start, interval.get(i2).end));
		return list;
	}

	public static List<Integer> spiralOrder(int[][] matrix) {
		List<Integer> list = new ArrayList<>();
		if (matrix.length == 0) {
			return list;
		}
		int rowBegin = 0;
		int rowEnd = matrix.length - 1;
		int colBegin = 0;
		int colEnd = matrix[0].length - 1;
		int count = 0;
		int nums = matrix.length * matrix[0].length;
		while (rowBegin <= rowEnd && colBegin <= colEnd) {
			for (int j = colBegin; j <= colEnd; j++) {
				list.add(matrix[rowBegin][j]);
			}
			rowBegin++;
			for (int i = rowBegin; i <= rowEnd; i++) {
				list.add(matrix[i][colEnd]);
			}
			colEnd--;
			if (rowBegin <= rowEnd && colBegin <= colEnd) {
				for (int j = colEnd; j >= colBegin; j--) {
					list.add(matrix[rowEnd][j]);
				}
			}
			rowEnd--;
			if (rowBegin <= rowEnd && colBegin <= colEnd) {
				for (int i = rowEnd; i >= rowBegin; i--) {
					list.add(matrix[i][colBegin]);
				}
			}
			colBegin++;
		}
		return list;
	}

	public static int maxSubArray(int[] nums) {
		int max = 0;
		int res = nums[0];
		for (int i = 0; i < nums.length; i++) {
			max += nums[i];
			if (max > res)
				res = max;
			if (max < 0)
				max = 0;
		}
		return res;
	}

	public static double myPow(double x, int n) {
		if (n < 0) {
			return 1 / pow(x, -n);
		}
		return pow(x, n);
	}

	public static double pow(double x, int n) {
		if (n == 0) {
			return 1;
		}
		double next = pow(x, n / 2);
		if (n % 2 == 0) {
			return next * next;
		} else {
			return x * next * next;
		}
	}

	public static List<List<Integer>> permuteUnique(int[] nums) {
		List<List<Integer>> list = new ArrayList<>();
		boolean[] used = new boolean[nums.length];
		List<Integer> al = new ArrayList<>();
		Arrays.sort(nums);
		backtracking(nums, list, al, used);
		return list;
	}

	public static void backtracking(int[] nums, List<List<Integer>> list, List<Integer> al, boolean[] used) {
		if (al.size() == nums.length) {
			list.add(new ArrayList(al));
		} else {
			for (int i = 0; i < nums.length; i++) {
				if (used[i])
					continue;
				if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])
					continue;
				used[i] = true;
				al.add(nums[i]);
				backtracking(nums, list, al, used);
				used[i] = false;
				al.remove(al.size() - 1);
			}
		}
	}

	public static String multiply(String num1, String num2) {
		char[] c1 = num1.toCharArray();
		char[] c2 = num2.toCharArray();
		int n1 = num1.length();
		int n2 = num2.length();
		int[] mul = new int[n1 + n2];
		int index1, index2, res;
		for (int i = n1 - 1; i >= 0; i--) {
			for (int j = n2 - 1; j >= 0; j--) {
				index1 = i + j;
				index2 = i + j + 1;
				res = (c1[i] - '0') * (c2[j] - '0') + mul[index2];
				mul[index2] = res % 10;
				mul[index1] += res / 10;
			}
		}
		StringBuilder sb = new StringBuilder();
		if (mul.length == 1) {
			sb.append(mul[0]);
		}
		if (mul[0] != 0) {
			sb.append(mul[0]);
		}
		for (int i = 1; i < n1 + n2; i++) {
			sb.append(mul[i]);
		}
		String s = sb.toString();
		while (s.charAt(0) == 0 && s.length() != 1) {
			s = s.substring(1);
		}
		return s;
	}

	public static int jump(int[] nums) {
		int a = nums.length;
		if (a < 2)
			return 0;
		int currentMax = 0, nextMax = 0, level = 0;
		for (int i = 0; i < a - 1; i++) {
			nextMax = Math.max(nextMax, i + nums[i]);
			if (currentMax == i) {
				level++;
				currentMax = nextMax;
			}
		}
		return level;
	}

	public static List<List<Integer>> permute(int[] nums) {
		List<List<Integer>> all = new ArrayList<>();
		ArrayList<Integer> al = new ArrayList<>();
		boolean[] used = new boolean[nums.length];
		dfs(nums, used, al, all);
		return all;
	}

	public static void dfs(int[] nums, boolean[] used, ArrayList<Integer> al, List<List<Integer>> all) {
		if (al.size() == nums.length) {
			ArrayList<Integer> al2 = new ArrayList<>(al);
			all.add(al2);
		} else {
			for (int i = 0; i < nums.length; i++) {
				if (used[i])
					continue;
				used[i] = true;
				al.add(nums[i]);
				dfs(nums, used, al, all);
				al.remove(al.size() - 1);
				used[i] = false;
			}
		}
	}

	public static int reverse(int x) {
		int res = 0;
		boolean flag = true;
		if (x < 0) {
			flag = false;
			x = -x;
		}
		int last;
		while (x > 0) {
			last = x % 10;
			res = res * 10 + last;
			x /= 10;
		}

		return flag ? res : -res;
	}
}